from obspy.clients.fdsn.mass_downloader.domain import (
    Domain,
    CircularDomain,
    RectangularDomain,
)
from obspy.geodetics.base import locations2degrees


class RectangleDomain(RectangularDomain):
    """
    A rectangular domain defined by latitude and longitude bounds. Edges are included in the domain.

    :param minlatitude: Minimum latitude
    :type minlatitude: float
    :param maxlatitude: Maximum latitude
    :type maxlatitude: float
    :param minlongitude: Minimum longitude
    :type minlongitude: float
    :param maxlongitude: Maximum longitude
    :type maxlongitude: float
    """

    def __init__(self, minlatitude, maxlatitude, minlongitude, maxlongitude):
        super().__init__(minlatitude, maxlatitude, minlongitude, maxlongitude)
        self.minlatitude = minlatitude
        self.maxlatitude = maxlatitude
        self.minlongitude = minlongitude
        self.maxlongitude = maxlongitude

    def is_in_domain(self, latitude, longitude):
        """
        Checks whether a point is within the domain

        :param latitude: Latitude of query point
        :type latitude: float
        :param longitude: Longitude of query point
        :type longitude: float
        :return: True if point is within the domain, false otherwise
        :rtype: bool
        """
        return (
            self.minlatitude <= latitude <= self.maxlatitude
            and self.minlongitude <= longitude <= self.maxlongitude
        )


class CircleDomain(CircularDomain):
    """
    Circular domain for selecting coordinates within a given radii of sourcepoint.
    The edges are not included in the domain

    :param latitude: Latitude of the circle center
    :type latitude: float
    :param longitude: Longitude of the circle center
    :type longitude: float
    :param minradius: Minimum radius in degrees
    :type minradius: float
    :param maxradius: maximum radius in degrees
    :type maxradius: float
    """

    def __init__(self, latitude, longitude, minradius, maxradius):
        super().__init__(latitude, longitude, minradius, maxradius)

    def is_in_domain(self, latitude, longitude):
        """
        Checks whether a point is within the domain

        :param latitude: Latitude of query point
        :type latitude: float
        :param longitude: Longitude of query point
        :type longitude: float
        :return: True if point is within the domain, false otherwise
        :rtype: bool
        """
        d = locations2degrees(self.latitude, self.longitude, latitude, longitude)
        return self.minradius < d < self.maxradius


class Germany(Domain):
    """
    Example usage of how to create more complex region geometries.
    https://docs.obspy.org/_modules/obspy/clients/fdsn/mass_downloader/domain.html
    """

    def __init__(self):
        Domain.__init__(self)
        try:
            import fiona
            import shapely.geometry
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "The Germany domain requires fiona and shapely. "
                "Please install fiona and shapely, e.g., using pip."
            )

        fiona_collection = fiona.open("./shape_files/DEU_adm/DEU_adm0.shp")
        geometry = fiona_collection.next()["geometry"]
        self.shape = shapely.geometry.asShape(geometry)
        self.b = fiona_collection.bounds

    def get_query_parameters(self):
        return {
            "minlatitude": self.b[1],
            "minlongitude": self.b[0],
            "maxlatitude": self.b[3],
            "maxlongitude": self.b[2],
        }

    def is_in_domain(self, latitude, longitude):
        """
        Checks whether a point is within the domain

        :param latitude: Latitude of query point
        :type latitude: float
        :param longitude: Longitude of query point
        :type longitude: float
        :return: True if point is within the domain, false otherwise
        :rtype: bool
        """
        return self.shape.contains(shapely.geometry.Point(longitude, latitude))
