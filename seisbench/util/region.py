from obspy.clients.fdsn.mass_downloader.domain import (
    Domain,
    CircularDomain,
    RectangularDomain,
)
import fiona
import shapely.geometry


class RectangleDomain(RectangularDomain):
    """
    A rectangular domain defined by latitude and longitude bounds.
    """

    def __init__(self, minlatitude, maxlatitude, minlongitude, maxlongitude):
        self.minlatitude = minlatitude
        self.maxlatitude = maxlatitude
        self.minlongitude = minlongitude
        self.maxlongitude = maxlongitude

    def is_in_domain(self, latitude, longitude):
        if (
            latitude >= self.minlatitude
            and latitude <= self.maxlatitude
            and longitude >= self.minlongitude
            and longitude <= self.maxlongitude
        ):
            return True
        return False


class CircleDomain(CircularDomain):
    """
    Circular domain for selecting coordinates within a given radii of sourcepoint.
    """

    def __init__(self, latitude, longitude, minradius, maxradius):
        super().__init__(latitude, longitude, minradius, maxradius)

    def is_in_domain(self, latitude, longitude):
        d = ((self.latitude - latitude) ** 2 + (self.longitude - longitude) ** 2) ** 0.5
        if d < self.maxradius and d > self.minradius:
            return True
        return False


class Germany(Domain):
    """
    Example usage of how to create more complex region geometries.
    https://docs.obspy.org/_modules/obspy/clients/fdsn/mass_downloader/domain.html
    """

    def __init__(self):
        Domain.__init__(self)
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
        if self.shape.contains(shapely.geometry.Point(longitude, latitude)):
            return True
        return False
