
from numpy import searchsorted


class Slice(object):
    """
    Mixin for extracting a single column of a Term's output.

    Parameters
    ----------
    term : zipline.pipeline.term.Term
        The term from which to extract a column of data.
    asset : zipline.assets.Asset
        The asset corresponding to the column of `term` to be extracted.

    Notes
    -----
    Users should rarely construct instances of `Slice` directly. Instead, they
    should construct instances via indexing, e.g. `MyFactor()[Asset(24)]`.
    """

    def __new__(cls, term, asset):
        return super(Slice, cls).__new__(
            cls,
            asset=asset,
            inputs=[term],
            window_length=0,
            mask=term.mask,
            dtype=term.dtype,
            missing_value=term.missing_value,
            window_safe=term.window_safe,
            ndim=1,
        )

    def __repr__(self):
        return "{type}({parent_term}, column={asset})".format(
            type=type(self).__name__,
            parent_term=type(self.inputs[0]).__name__,
            asset=self._asset,
        )

    def _init(self, asset, *args, **kwargs):
        self._asset = asset
        return super(Slice, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, asset, *args, **kwargs):
        return (super(Slice, cls)._static_identity(*args, **kwargs), asset)

    def _compute(self, windows, dates, assets, mask):
        asset_column = searchsorted(assets.values, self._asset.sid)
        col = windows[0][:, asset_column]
        # Return a 2D array with one column rather than a 1D array of the
        # column.
        return col[:, None]
