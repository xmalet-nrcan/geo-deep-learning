

class SatellitePass(Enum):
    ASCENDING = "Ascending"
    DESCENDING = "Descending"

    @classmethod
    def from_str(cls, s: str) -> "SatellitePass":
        """Convert a string to a SatellitePass enum."""
        translate_dict = {
            "A": "Ascending",
            "D": "Descending",
            "ASC": "Ascending",
            "DESC": "Descending",
            "ASCENDING": "Ascending",
            "DESCENDING": "Descending",
        }
        try:
            return cls[translate_dict[s.upper()].upper()]
        except KeyError:
            raise ValueError(f"Satellite pass {s} not recognized.")


class BandName(Enum):
    BITMASK_CROPPED = 1
    LOCALINCANGLE = 2
    M = 3
    NDSV = 4
    PDN = 5
    PSN = 6
    PVN = 7
    RFDI = 8
    RL = 9
    RR = 10
    S0 = 11
    SP1 = 12
    SP2 = 13
    SP3 = 14


def band_names_to_indices(band_names: Optional[List[Any]]) -> Optional[List[int]]:
    """
    Convert a list of band names (str or BandName) into indices (int) according to BandName.

    Accepts ["RR", "RL", "M"] or [BandName.RR, BandName.RL, BandName.M].
    """
    if band_names is None:
        return None

    indices = []
    for name in band_names:
        if isinstance(name, BandName):
            indices.append(name.value)
        elif isinstance(name, str):
            try:
                indices.append(BandName[name].value)
            except KeyError:
                raise ValueError(f"Unknown band name: {name}")
        else:
            raise TypeError(f"Unsupported type for band_names: {type(name)}")
    return indices
