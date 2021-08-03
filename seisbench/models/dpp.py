from .base import WaveformPipeline, WaveformModel, CustomLSTM, ActivationLSTMCell

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
import tqdm


class DeepPhasePick(WaveformPipeline):
    """
    Note: ...

    """

    def __init__(
        self,
        **kwargs,
    ):
        # TODO: update
        citation = (
            "Soto, H. & Schurr, B. (2021). "
            "DeepPhasePick: A method for Detecting and Picking SeismicPhases from Local "
            "Earthquakes based on highly optimized Convolutional and Recurrent Deep Neural Networks. "
            "EarthArXiv. https://doi.org/10.31223/X5BC8B"
            "Geophys. J. Int. https://doi.org/10.1093/gji/ggab266"
        )
        super().__init__(
            citation=citation,
            **kwargs,
        )

        # self.in_channels = in_channels
        # self.classes = classes
        # self._phases = phases
        # self.original_compatible = original_compatible
        # if phases is not None and len(phases) != classes:
        #     raise ValueError(
        #         f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
        #     )


class DPPDetector(WaveformModel):

    def __init__(
        self,
        in_channels=3,
        nclasses=3,
        phases=None,
        sampling_rate=100,
        pred_sample=200,
        original_compatible=False,
        **kwargs,
    ):
        super().__init__(
            output_type="point",
            in_samples=480, # TODO: in principle should be different based on optimized hyperparameter, alghouth all available models so far use this hyperparameter value
            pred_sample=pred_sample,
            labels=phases,
            sampling_rate=sampling_rate,
            default_args={
                "stride": 10,
                "op_conds": ['1', '2', '3', '4'],
                "tp_th_add": 1.5,
                "dt_sp_near": 2.,
                "dt_ps_max": 35.,
                "dt_sdup_max": 2.,
                },
            **kwargs,
        )

        # TODO: check if momentum is the same than in keras
        #
        self.in_channels = input_channels
        self.classes = nclasses
        self.stride = 1

        # # self.original_compatible = True
        # self.original_compatible = False

        # groups == in_channels in Conv1d is for “depthwise convolution”, equivalent to keras SeparableConv1D layer.

        self.conv1 = nn.Conv1d(
            self.in_channels,
            12,
            17,
            self.stride,
            padding=17 // 2,
            groups=self.in_channels,
        )
        self.bn1 = nn.BatchNorm1d(12, eps=1e-3, momentum=0.99)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(12, 24, 11, self.stride, padding=11 // 2, groups=12)
        self.bn2 = nn.BatchNorm1d(24, eps=1e-3, momentum=0.99)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv1d(24, 48, 5, self.stride, padding=5 // 2, groups=24)
        self.bn3 = nn.BatchNorm1d(48, eps=1e-3, momentum=0.99)
        self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(48, 96, 9, self.stride, padding=9 // 2, groups=48)
        self.bn4 = nn.BatchNorm1d(96, eps=1e-3, momentum=0.99)
        self.dropout4 = nn.Dropout(0.4)
        self.conv5 = nn.Conv1d(96, 192, 17, self.stride, padding=17 // 2, groups=96)
        self.bn5 = nn.BatchNorm1d(192, eps=1e-3, momentum=0.99)
        self.dropout5 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(2880, 50)
        self.bn6 = nn.BatchNorm1d(50, eps=1e-3, momentum=0.99)
        self.dropout6 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, self.classes)

        self.pool = nn.MaxPool1d(2, 2)

        self.activation1 = torch.relu
        self.activation2 = torch.sigmoid
        self.activation3 = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x = self.bn1(self.activation1((self.conv1(x))))
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.bn2(self.activation1((self.conv2(x))))
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.bn3(self.activation1((self.conv3(x))))
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.bn4(self.activation1((self.conv4(x))))
        x = self.pool(x)
        x = self.dropout4(x)

        x = self.bn5(self.activation2((self.conv5(x))))
        x = self.pool(x)
        x = self.dropout5(x)

        x = torch.flatten(x, 1)

        x = self.bn6(self.activation1(self.fc1(x)))
        x = self.dropout6(x)
        x = self.fc2(x)
        x = self.activation3(x)

        return x

    def classify_aggregate(self, annotations, argdict):
        """
        Converts the annotations to discrete picks using :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = []
        for phase in self.phases:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"DPP_{phase}"),
                argdict.get(f"{phase}_threshold", 0.9),
                phase,
            )

        picks_out = _select_picks(self, annotations, picks, argdict)

        return sorted(picks_out)


    def _select_picks(self, annotations, picks, argdict):
        """
        Applies optional conditions to improve phase detection. Some preliminary picks are removed or kept depending on these conditions.

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :param picks: List of picks returned by :py:func:`classify_aggregate` function.
        :return: List of selected picks
        """
        # user-defined parameters
        op_conds = argdict['op_conds'] # optional conditions
        tp_th_add = argdict['tp_th_add'] # seconds
        dt_sp_near = argdict['dt_sp_near'] # seconds
        dt_ps_max = argdict['dt_ps_max'] # seconds
        dt_sdup_max = argdict['dt_sdup_max'] # seconds
        #
        # TODO: integrate optimized hyperparameters, perhaps when initializing the class and pass them to argdict ??
        params = {'win_size': 480, 'frac_dsamp_p1': 0.7, 'frac_dsamp_s1': 0.5}
        tp_shift = (params['frac_dsamp_p1']-.5) * params['win_size'] * samp_dt
        ts_shift = (params['frac_dsamp_s1']-.5) * params['win_size'] * samp_dt
        picks_p = [pick for pick in picks if pick.phase == 'P']
        picks_s = [pick for pick in picks if pick.phase == 'S']
        #
        prob_P = annotations[0]
        prob_S = annotations[1]
        #
        pb_picks_p = [pick.peak_value for pick in picks_p]
        pb_picks_s = [pick.peak_value for pick in picks_s]
        tt_p_arg = [np.argmin(abs(prob_P.data - pb)) for pb in pb_picks_p]
        tt_s_arg = [np.argmin(abs(prob_S.data - pb)) for pb in pb_picks_s]
        tpicks_p = np.array([pick.peak_time.timestamp + tp_shift for pick in picks_p])
        tpicks_s = np.array([pick.peak_time.timestamp + ts_shift for pick in picks_s])
        print(f"triggered picks (P, S): {len(picks_p)}, {len(picks_s)}")
        #
        p_picks_bool = np.full(len(tpicks_p), True)
        s_picks_bool = np.full(len(tpicks_s), True)
        p_arg_selected = np.where(p_picks_bool)[0]
        s_arg_selected = np.where(s_picks_bool)[0]
        s_arg_used = []
        #
        samp_freq = self.sampling_rate
        samp_dt = 1 / samp_freq
        #
        # (1) Iterate over predicted P picks, in order to resolve between P and S phases predicted close in time, with overlapping probability time series
        #
        if '1' in op_conds:
            #
            for i, tp in enumerate(tpicks_p[:]):
                #
                # search S picks detected nearby P phases
                #
                cond_pre = prob_P[:tt_p_arg[i]] > .5
                cond_pre = cond_pre[::-1]
                if len(cond_pre) > 0:
                    tp_th_pre = tp - (np.argmin(cond_pre) * argdict['stride'] * samp_dt) - tp_th_add
                else:
                    tp_th_pre = tp - tp_th_add
                #
                cond_pos = prob_P[tt_p_arg[i]:] > .5
                tp_th_pos = tp + (np.argmin(cond_pos) * argdict['stride'] * samp_dt) + tp_th_add
                #
                ts_in_th = [(t, tss) for t, tss in enumerate(tpicks_s) if tss >= tp_th_pre and tss <= tp_th_pos]
                #
                # picks detected before and after current P pick
                #
                # tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_p) if tpp > tp - dt_ps_max and tpp < tp_th_pre]
                # tp_in_next = [(t, tpp) for t, tpp in enumerate(tpicks_p) if tpp >= tp_th_pos and tpp <= tp + dt_ps_max]
                # ts_in_next = [(t, tss) for t, tss in enumerate(tpicks_s) if tss >= tp_th_pos and tss <= tp + dt_ps_max]
                #
                if len(ts_in_th) > 0:
                    #
                    # pick = P/S or S/P
                    s_arg_used.append(ts_in_th[0][0])
                    #
                    if prob_P[tt_p_arg[i]] >= prob_S[tt_s_arg[ts_in_th[0][0]]]:
                        #
                        # P kept, S discarded
                        s_picks_bool[ts_in_th[0][0]] = False
                    else:
                        #
                        # S kept, P discarded
                        p_picks_bool[i] = False
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (2) iterate over selected S picks in order to resolve between P and S phases predicted close in time, with non-overlapping probability time series
        #
        if '2' in op_conds:
            #
            dct_sp_near = {}
            for i, s_arg in enumerate(s_arg_selected):
                #
                dct_sp_near[s_arg] = []
                ts = tpicks_s[s_arg]
                #
                s_cond_pos = prob_S[tt_s_arg[s_arg]:] > .5
                ts_th_pos = ts + (np.argmin(s_cond_pos) * argdict['stride'] * samp_dt)
                #
                s_cond_pre = prob_S[:tt_s_arg[s_arg]] > .5
                s_cond_pre = s_cond_pre[::-1]
                ts_th_pre = ts - (np.argmin(s_cond_pre) * argdict['stride'] * samp_dt)
                #
                for j, p_arg in enumerate(p_arg_selected):
                    #
                    tp = tpicks_p[p_arg]
                    #
                    p_cond_pos = prob_P[tt_p_arg[p_arg]:] > .5
                    tp_th_pos = tp + (np.argmin(p_cond_pos) * argdict['stride'] * samp_dt)
                    #
                    p_cond_pre = prob_P[:tt_p_arg[p_arg]] > .5
                    p_cond_pre = p_cond_pre[::-1]
                    # tp_th_pre = tp - (np.argmin(p_cond_pre) * argdict['stride'] * samp_dt)
                    if len(p_cond_pre) > 0:
                        tp_th_pre = tp - (np.argmin(p_cond_pre) * argdict['stride'] * samp_dt)
                    else:
                        tp_th_pre = tp
                    #
                    dt_sp_th = abs(ts_th_pos - tp_th_pre)
                    dt_ps_th = abs(tp_th_pos - ts_th_pre)
                    #
                    if dt_sp_th < dt_sp_near or dt_ps_th < dt_sp_near:
                        dct_sp_near[s_arg].append([p_arg, min(dt_sp_th, dt_ps_th)])
            #
            # for possible nearby P/S phases, presumed false ones are discarded
            for s_arg in dct_sp_near:
                if len(dct_sp_near[s_arg]) > 0:
                    #
                    pb_s_near = prob_S[tt_s_arg[s_arg]]
                    pb_p_near_arg = np.argmin([p_near[1] for p_near in dct_sp_near[s_arg]])
                    p_near_arg = dct_sp_near[s_arg][pb_p_near_arg][0]
                    pb_p_near = prob_P[tt_p_arg[p_near_arg]]
                    #
                    if pb_s_near >= pb_p_near:
                        p_picks_bool[p_near_arg] = False
                    else:
                        s_picks_bool[s_arg] = False
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (3) iterate over selected S picks. S picks for which there is no earlier P or P-S predicted picks will be discarded
        #
        if '3' in op_conds:
            #
            for i, s_arg in enumerate(s_arg_selected):
                #
                ts = tpicks_s[s_arg]
                #
                # P picks detected before current S pick
                #
                tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_p) if tpp > ts - dt_ps_max and tpp < ts and p_picks_bool[t]]
                #
                if len(tp_in_prior) == 0:
                    #
                    # prior pick not found --> discard
                    s_picks_bool[s_arg] = False
                #
                if len(tp_in_prior) > 0:
                    #
                    tp_prior = tp_in_prior[-1][1]
                    ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_s) if tss > tp_prior and tss < ts and t in np.where(s_picks_bool)[0]]
                    #
                    if len(ts_in_prior) > 1:
                        s_picks_bool[s_arg] = False
                    #
                    # if len(ts_in_prior) == 1:
                    #     #
                    #     ts_prior = ts_in_prior[0][1]
                    #     if ts > ts_prior + abs(tp_prior - ts_prior):
                    #         s_picks_bool[i] = False
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (4) iterate over selected S picks in order to resolve between possible duplicated S phases
        #
        if '4' in op_conds:
            #
            s_arg_used_dup = []
            dct_s_dup = {}
            for i, s_arg in enumerate(s_arg_selected):
                #
                dct_s_dup[s_arg] = [s_arg]
                ts = tpicks_s[s_arg]
                cond_pos = prob_S[tt_s_arg[s_arg]:] > .5
                ts_th_pos = ts + (np.argmin(cond_pos) * argdict['stride'] * samp_dt)
                #
                for j, s_arg2 in enumerate(s_arg_selected[i+1: len(s_arg_selected)]):
                    #
                    ts2 = tpicks_s[s_arg2]
                    cond_pre = prob_S[:tt_s_arg[s_arg2]] > .5
                    cond_pre = cond_pre[::-1]
                    ts2_th_pre = ts2 - (np.argmin(cond_pre) * argdict['stride'] * samp_dt)
                    #
                    if abs(ts_th_pos - ts2_th_pre) < dt_sdup_max:
                        dct_s_dup[s_arg].append(s_arg2)
                    else:
                        break
            #
            # for possible duplicated S phases, presumed false ones are discarded
            for s_arg in dct_s_dup:
                if len(dct_s_dup[s_arg]) > 1:
                    pb_s_dup = np.array([prob_S[tt_s_arg[s_arg_dup]] for s_arg_dup in dct_s_dup[s_arg]])
                    pb_s_dup_argmax = np.argmax(pb_s_dup)
                    s_arg_false = [s_arg3 for s_arg3 in dct_s_dup[s_arg] if s_arg3 != dct_s_dup[s_arg][pb_s_dup_argmax]]
                    for s_false in s_arg_false:
                        s_picks_bool[s_false] = False
                        s_arg_used_dup.append(s_false)
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # selected picks
        #
        print(f"selected picks (P, S): {len(np.where(p_picks_bool)[0])}, {len(np.where(s_picks_bool)[0])}")
        #
        picks_p_out = [pick for i, pick in enumerate(picks_p) if p_picks_bool[i]]
        picks_s_out = [pick for i, pick in enumerate(picks_s) if s_picks_bool[i]]
        picks_out = picks_p_out + picks_s_out
        #
        return picks_out




class DPPPicker(WaveformModel):

    # TODO: update
    def __init__(
        self,
        in_channels=3,
        phases=None,
        sampling_rate=100,
        **kwargs,
    ):
        super().__init__(
            sampling_rate=sampling_rate,
            labels=phases,
            default_args={"mcd_iter": 10},
            **kwargs,
        )


        self.mode = labels # TODO: check

        if self.mode == "P":
            self.lstm1 = CustomLSTM(
                ActivationLSTMCell, 1, 100, bidirectional=True, recurrent_dropout=0.2
            )
            self.lstm2 = CustomLSTM(
                ActivationLSTMCell, 200, 160, bidirectional=True, recurrent_dropout=0.25
            )
            self.dropout1 = nn.Dropout(0.2)
            self.dropout2 = nn.Dropout(0.35)
            self.fc1 = nn.Linear(320, 1)
        elif self.mode == "S":
            self.lstm1 = CustomLSTM(
                ActivationLSTMCell, 2, 20, bidirectional=True, recurrent_dropout=0.35
            )
            self.lstm2 = CustomLSTM(
                ActivationLSTMCell, 40, 30, bidirectional=True, recurrent_dropout=0.25
            )
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.45)
            self.fc1 = nn.Linear(60, 1)

        self.activation = torch.sigmoid

    def forward(self, x):
        # Permute shapes to match LSTM  --> (seq, batch, channels)

        x = x.permute(2, 0, 1)  # (batch, channels, seq) --> (seq, batch, channels)
        x = self.lstm1(x)[0]
        x = self.dropout1(x)
        x = self.lstm2(x)[0]
        x = self.dropout2(x)
        x = x.permute(1, 0, 2)

        # keras TimeDistributed layer is applied by:
        # -> reshaping from (batch, sequence, *) to (batch * sequence, *)
        # -> then applying the layer,
        # -> then reshaping back to (batch, sequence, *)
        #
        shape_save = x.shape
        x = x.reshape((-1,) + x.shape[2:])

        x = self.activation(self.fc1(x))
        x = x.reshape(shape_save[:2] + (1,))
        x = x.squeeze(-1)

        return x

    # version of annotate function specialized for picking models
    def annotate(
        self, stream, annotations, picks_in, strict=True, flexible_horizontal_components=True, **kwargs
    ):
        """
        Annotates an obspy stream using the model based on the configuration of the WaveformModel superclass.
        For example, for a picking model, annotate will give a characteristic function/probability function for picks
        over time.
        The annotate function contains multiple subfunction, which can be overwritten individually by inheriting
        models to accomodate their requirements. These functions are:

        - :py:func:`annotate_stream_pre`
        - :py:func:`annotate_stream_validate`
        - :py:func:`annotate_window_pre`
        - :py:func:`annotate_window_post`

        Please see the respective documentation for details on their functionality, inputs and outputs.

        :param stream: Obspy stream to annotate
        :type stream: obspy.core.Stream
        :param strict: If true, only annotate if recordings for all components are available,
                       otherwise impute missing data with zeros.
        :type strict: bool
        :param flexible_horizontal_components: If true, accepts traces with Z12 components as ZNE and vice versa.
                                               This is usually acceptable for rotationally invariant models,
                                               e.g., most picking models.
        :type flexible_horizontal_components: bool
        :param kwargs:
        :return: Obspy stream of annotations
        """
        # if self._annotate_function is None:
        #     raise NotImplementedError(
        #         "This model has no annotate function implemented."
        #     )

        # Kwargs overwrite default args
        argdict = self.default_args.copy()
        argdict.update(kwargs)

        stream = stream.copy()
        stream.merge(-1)

        output = []
        # output = obspy.Stream()
        # if len(stream) == 0:
        #     return output

        # Preprocess stream, e.g., filter/resample
        self.annotate_stream_pre(stream, argdict)

        # Validate stream
        self.annotate_stream_validate(stream, argdict)

        # Group stream
        groups = self.groups_stream_by_instrument(stream)

        # Stream to arrays to windows
        for group in groups:
            trace = group[0]
            trace_id = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"

            # Sampling rate of the data. Equal to self.sampling_rate is this is not None
            argdict["sampling_rate"] = trace.stats.sampling_rate # TODO: check

            picks_group = [pick for pick in picks_in if pick.trace_id == trace_id]
            for pick_in in picks_group:
                output.append(obspy.Stream())

                # TODO: create function stream_to_arrays() ...DONE
                # returns picking windows (data)
                times, data = self.stream_to_arrays(
                    group,
                    pick_in,
                    strict=strict,
                    flexible_horizontal_components=flexible_horizontal_components,
                )

                # TODO: create function _annotate_window(), instead of_annotate_point() ...DONE
                # -> model prediction needs to be adapted to pytorch
                #
                # runs Monte Carlo Dropout (MCD) on picking windows, returning mc_pred array (as preds) containing results from MCD iteration
                pred_times, pred_rates, preds = self._annotate_window(
                    times, data, pick_in, argdict
                )

                # Write to output stream.
                # Returns streams containing results from MCD iteration.
                output[-1] += self._predictions_to_stream(pred_rates, pred_times, preds, pick_in, trace)

        return output

    def _predictions_to_stream(self, pred_rates, pred_times, preds, pick_in, trace):
        """
        Converts a set of predictions to obspy streams

        :param pred_rates: Sampling rates of the prediction arrays
        :param pred_times: Start time of each prediction array
        :param preds: The prediction arrays, each with shape (samples, channels)
        :param trace: A source trace to extract trace naming from
        :return: Obspy stream of predictions
        """
        output = obspy.Stream()
        for (pred_time, pred_rate, pred) in zip(pred_times, pred_rates, preds):
            for i in range(pred.shape[1]):
                label = f"{pick_in.phase}"i

                trimmed_pred, f, _ = self._trim_nan(pred[:, i])
                trimmed_start = pred_time + f / pred_rate
                output.append(
                    obspy.Trace(
                        trimmed_pred,
                        {
                            "starttime": trimmed_start,
                            "sampling_rate": pred_rate,
                            "network": trace.stats.network,
                            "station": trace.stats.station,
                            "location": trace.stats.location,
                            "channel": f"{self.__class__.__name__}_{label}",
                        },
                    )
                )

        return output

    # runs Monte Carlo Dropout (MCD) on picking windows, returning mc_pred array (as preds) containing results from MCD iteration
    def _annotate_window(self, times, data, pick_in, argdict):
        """
        Annotation function for a point prediction model using a sliding window approach.
        Will use the key `stride` from the `argdict` to determine the shift (in samples) between two windows.
        Default `stride` is 1.
        This function expects model outputs after postprocessing for each window to be scalar or 1D arrays.
        """
        pred_times = []
        pred_rates = []
        full_preds = []
        mc_iter = argdict['mcd_iter']

        # Iterate over all blocks of waveforms
        for t0, block in zip(times, data):
            mc_pred = []
            block /= np.abs(block).max() # normalize before predicting picks
            if pick_in.phase == 'P':
                block = block[:1] # pick on vertical component
                block = block.reshape(1, block.shape[1], 1)
            elif pick_in.phase == 'S':
                block = block[-2:] # pick on two horizontal components
                block = block.T.reshape(1, block.shape[1], 2)
            #
            fragment = torch.tensor(block, device=self.device, dtype=torch.float32)
            for j in tqdm.tqdm(range(mc_iter)):
                # x_mc = data.reshape(1, data.shape[1], data.shape[2])
                x_mc = data
                #
                # TODO: adapt to pytorch
                #
                y_mc = model['best_model'].predict(x_mc, batch_size=model['batch_size_pred'], verbose=0)
                # with torch.no_grad():
                #     preds = self._predict_and_postprocess_windows(argdict, fragments)
                #     ...
                # mc_pred.append(y_mc) # y_mc.shape = (1, block.shape[1], 1)
                mc_pred.append(np.random.rand(1, block.shape[1], 1)) # for testing
            #
            pred_times.append(t0)
            pred_rates.append(argdict["sampling_rate"])
            # mc_pred = np.array(mc_pred)[:,0,:,:] # mc_pred.shape = (mc_iter, block.shape[1], 1)
            mc_pred = np.array(mc_pred)[:,0,:,0].T # mc_pred.shape = (block.shape[1], mc_iter)
            full_preds.append(mc_pred)

        return pred_times, pred_rates, full_preds

    def stream_to_arrays(
        self, stream_in, pick_in, strict=True, flexible_horizontal_components=True
    ):
        """
        Converts streams into a list of start times and numpy arrays.
        Assumes:

        - All traces in the stream are from the same instrument and only differ in the components
        - No overlapping traces of the same component exist
        - All traces have the same sampling rate

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :param strict: If true, only if recordings for all components are available, otherwise impute missing data with zeros.
        :type strict: bool, default True
        :param flexible_horizontal_components: If true, accepts traces with Z12 components as ZNE and vice versa.
                                               This is usually acceptable for rotationally invariant models,
                                               e.g., most picking models.
        :type flexible_horizontal_components: bool
        :return: output_times: Start times for each array
        :return: output_data: Arrays with waveforms
        """
        seqnum = 0  # Obspy raises an error when trying to compare traces. The seqnum hack guarantees that no two tuples reach comparison of the traces.
        stream = stream_in.copy()
        if len(stream) == 0:
            return [], []

        sampling_rate = stream[0].stats.sampling_rate

        # streams to picking windows
        # TODO: integrate optimized hyperparameters, perhaps when initializing the class and pass them to argdict ??
        params = {'win_size': 480, 'frac_dsamp_p1': 0.7, 'frac_dsamp_s1': 0.5}
        samp_dt = 1. / sampling_rate
        if pick_in.phase == 'P':
            twd_1 = params['frac_dsamp_p1'] * params['win_size'] * samp_dt
        elif pick_in.phase == 'S':
            twd_1 = params['frac_dsamp_s1'] * params['win_size'] * samp_dt
        twd_2 = params['win_size'] * samp_dt - twd_1
        for tr in stream:
            tstart_win = pick_in.peak_time - twd_1
            tend_win = pick_in.peak_time + twd_2
            tr.trim(tstart_win, tend_win)

        component_order = self._component_order
        comp_dict = {c: i for i, c in enumerate(component_order)}

        matches = [
            ("1", "N"),
            ("2", "E"),
        ]  # Component regarded as identical if flexible_horizontal_components is True.
        if flexible_horizontal_components:
            for a, b in matches:
                if a in comp_dict:
                    comp_dict[b] = comp_dict[a]
                elif b in comp_dict:
                    comp_dict[a] = comp_dict[b]

        # Maps traces to the components existing. Allows to warn for mixed use of ZNE and Z12.
        existing_trace_components = defaultdict(list)

        start_sorted = PriorityQueue()
        for trace in stream:
            if trace.id[-1] in comp_dict and len(trace.data) > 0:
                start_sorted.put((trace.stats.starttime, seqnum, trace))
                seqnum += 1
                existing_trace_components[trace.id[:-1]].append(trace.id[-1])

        if flexible_horizontal_components:
            for trace, components in existing_trace_components.items():
                for a, b in matches:
                    if a in components and b in components:
                        seisbench.logger.warning(
                            f"Station {trace} has both {a} and {b} components. "
                            f"This might lead to undefined behavior. "
                            f"Please preselect the relevant components "
                            f"or set flexible_horizontal_components=False."
                        )

        active = (
            PriorityQueue()
        )  # Traces with starttime before the current time, but endtime after
        to_write = (
            []
        )  # Traces that are not active any more, but need to be written in the next array. Irrelevant for strict mode

        output_times = []
        output_data = []
        while True:
            if not start_sorted.empty():
                start_element = start_sorted.get()
            else:
                start_element = None
                if strict:
                    # In the strict case, all data would already have been written
                    break

            if not active.empty():
                end_element = active.get()
            else:
                end_element = None

            if start_element is None and end_element is None:
                # Processed all data
                break
            elif start_element is not None and end_element is None:
                active.put(
                    (start_element[2].stats.endtime, start_element[1], start_element[2])
                )
            elif start_element is None and end_element is not None:
                to_write.append(end_element[2])
            else:
                # both start_element and end_element are active
                if end_element[0] < start_element[0] or (
                    strict and end_element[0] == start_element[0]
                ):
                    to_write.append(end_element[2])
                    start_sorted.put(start_element)
                else:
                    active.put(
                        (
                            start_element[2].stats.endtime,
                            start_element[1],
                            start_element[2],
                        )
                    )
                    active.put(end_element)

            if not strict and active.qsize() == 0 and len(to_write) != 0:
                t0 = min(trace.stats.starttime for trace in to_write)
                t1 = max(trace.stats.endtime for trace in to_write)

                data = np.zeros(
                    (len(component_order), int((t1 - t0) * sampling_rate + 2))
                )  # +2 avoids fractional errors

                for trace in to_write:
                    p = int((trace.stats.starttime - t0) * sampling_rate)
                    cidx = comp_dict[trace.id[-1]]
                    data[cidx, p : p + len(trace.data)] = trace.data

                data = data[:, :-1]  # Remove fractional error +1

                output_times.append(t0)
                output_data.append(data)

                to_write = []

            if strict and active.qsize() == len(component_order):
                traces = []
                while not active.empty():
                    traces.append(active.get()[2])

                t0 = max(trace.stats.starttime for trace in traces)
                t1 = min(trace.stats.endtime for trace in traces)

                short_traces = [trace.slice(t0, t1) for trace in traces]
                data = np.zeros(
                    (len(component_order), len(short_traces[0].data) + 2)
                )  # +2 avoids fractional errors
                for trace in short_traces:
                    cidx = comp_dict[trace.id[-1]]
                    data[cidx, : len(trace.data)] = trace.data

                data = data[:, :-2]  # Remove fractional error +2

                output_times.append(t0)
                output_data.append(data)

                for trace in traces:
                    if t1 < trace.stats.endtime:
                        start_sorted.put((t1, seqnum, trace.slice(starttime=t1)))
                        seqnum += 1

        return output_times, output_data

    def classify_aggregate(self, annotations, picks_in, argdict):
        """
        Converts the annotations to discrete picks using :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = []
        for annotation in annotations:
            phase = annotation[0].stats.channel.split('_')[-1][0]
            #
            # TODO: create function picks_from_annotations() ...DONE
            # uses annotations (streams formed by array of MCD results) to obtain picks and their statistics (uncertaintiy, etc)
            picks += self.picks_from_annotations(
                annotation),
                argdict,
                phase,
            )

        return sorted(picks)

    @staticmethod
    def picks_from_annotations(annotation, argdict, phase):
        """
        Converts the annotations streams for a single phase to discrete picks using a classical trigger on/off.
        The lower threshold is set to half the higher threshold.
        Picks are represented by :py:class:`~seisbench.util.annotations.Pick` objects.
        The pick start_time and end_time are set to the trigger on and off times.

        :param annotations: Stream of annotations
        :param threshold: Higher threshold for trigger
        :param phase: Phase to label, only relevant for output phase labelling
        :return: List of picks
        """
        # picks = []
        trace_id = (
            f"{annotation[0].stats.network}.{annotation[0].stats.station}.{annotation[0].stats.location}"
        )
        t0 = annotation[0].stats.starttime
        t1 = annotation[0].stats.endtime

        # retrieve MCD predictions from annotation
        mc_pred = []
        for trace in annotation:
            mc_pred.append(trace.data)
        mc_pred = np.array(mc_pred)
        #
        mc_pred_mean = mc_pred.mean(axis=0)
        mc_pred_mean_class = (mc_pred_mean > .5).astype('int32')
        mc_pred_mean_arg_pick = mc_pred_mean_class.argmax(axis=0)[0]
        mc_pred_mean_tpick = mc_pred_mean_arg_pick / argdict['sampling_rate']
        mc_pred_std = mc_pred.std(axis=0)
        mc_pred_std_pick = mc_pred_std[mc_pred_mean_arg_pick][0]

        # calculate tpick uncertainty from std of mean probability
        prob_th1 = mc_pred_mean[mc_pred_mean_arg_pick,0] - mc_pred_std_pick
        prob_th2 = mc_pred_mean[mc_pred_mean_arg_pick,0] + mc_pred_std_pick
        cond = (mc_pred_mean > prob_th1) & (mc_pred_mean < prob_th2)
        samps_th = np.arange(mc_pred_mean.shape[0])[cond[:,0]]

        # this restricts the uncertainty calculation to the time interval between the predicted time onset (mc_pred_mean_tpick) and the first intersections
        # (in the rare case that these are not unique) of the mean probability (mc_pred_mean) with prob_th1 (before the onset) and with prob_th2 (after the onset)
        try:
            samps_th1 = np.array([s for s, samp in enumerate(samps_th[:]) if (samp < mc_pred_mean_arg_pick) and (samps_th[s+1] - samp > 1)]).max()
        except ValueError:
            samps_th1 = -1
        try:
            samps_th2 = np.array([s for s, samp in enumerate(samps_th[:-1]) if (samp > mc_pred_mean_arg_pick) and (samps_th[s+1] - samp > 1)]).min()
        except ValueError:
            samps_th2 = len(samps_th)

        samps_th = samps_th[samps_th1+1: samps_th2+1]
        mc_pred_mean_tpick_th1 = samps_th[0] / argdict['sampling_rate']
        mc_pred_mean_tpick_th2 = samps_th[-1] / argdict['samp_freq']
        # mc_pred_mean_tres = tpick_det - mc_pred_mean_tpick

        # pick class
        terr_pre = abs(mc_pred_mean_tpick - mc_pred_mean_tpick_th1)
        terr_pos = abs(mc_pred_mean_tpick - mc_pred_mean_tpick_th2)
        terr_mean = (terr_pre + terr_pos) * .5
        pick_class = 3
        if terr_mean <= .2:
            pick_class -= 1
        if terr_mean <= .1:
            pick_class -= 1
        if terr_mean <= .05:
            pick_class -= 1

        pick = util.Pick(
            trace_id=trace_id,
            start_time=t0,
            end_time=t1,
            peak_time=mc_pred_mean_tpick,
            peak_value=mc_pred_mean_arg_pick,
            peak_uncertainty=[terr_pre, terr_pos],
            peak_class=pick_class,
            phase=phase,
        )

        return pick


# ~seisbench.util.annotations.Pick adapted to DPPPicker
class Pick:
    """
    This class serves as container for storing pick information.
    Defines an ordering based on start time, end time and trace id.

    :param trace_id: Id of the trace the pick was generated from
    :param start_time: Onset time of the pick
    :param end_time: End time of the pick
    :param peak_time: Peak time of the characteristic function for the pick
    :param peak_value: Peak value of the characteristic function (mean of the MCD class probabilities) at the pick time
    :param peak_uncertainty: Peak time uncertainty for the pick. List of two values giving the uncertainty in seconds [before, after] the pick
    :param peak_class: Peak class derived from weighting scheme applied to pick time uncertainty
    :param phase: Phase hint
    """

    def __init__(
        self,
        trace_id,
        start_time,
        end_time=None,
        peak_time=None,
        peak_value=None,
        peak_uncertainty=None,
        peak_class=None,
        phase=None,
    ):
        self.trace_id = trace_id
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.peak_value = peak_value
        self.peak_uncertainty = peak_uncertainty
        self.peak_class = peak_class
        self.phase = phase

        if end_time is not None and peak_time is not None:
            if not start_time <= peak_time <= end_time:
                raise ValueError("peak_time must be between start_time and end_time.")

    def __lt__(self, other):
        if self.start_time < other.start_time:
            return True
        if self.end_time < other.end_time:
            return True
        if self.trace_id < other.trace_id:
            return True
        return False
