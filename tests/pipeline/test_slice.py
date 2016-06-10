"""
Tests for slicing pipeline terms.
"""
from numpy import arange
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
    Timestamp,
)
from pandas.util.testing import assert_frame_equal

from zipline.errors import NonWindowSafeInput, UnsupportedPipelineColumn
from zipline.pipeline import CustomFactor, Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors import (
    FactorSlice,
    Returns,
    RollingLinearRegressionOfReturns,
    RollingPearsonOfReturns,
    RollingSpearmanOfReturns,
    SimpleMovingAverage,
)
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.testing import check_arrays, parameter_space
from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase

from .test_engine import (
    AssetID,
    AssetIDPlusDay,
    OpenPrice,
)


class SliceTestCase(WithTradingEnvironment, ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(SliceTestCase, cls).init_class_fixtures()

        day = cls.env.trading_day
        sids = cls.sids

        cls.dates = dates = date_range(
            '2015-02-01', '2015-02-28', freq=day, tz='UTC',
        )

        cls.raw_data = DataFrame(
            data=arange(len(dates) * len(sids), dtype=float).reshape(
                len(dates), len(sids),
            ),
            index=dates,
            columns=cls.asset_finder.retrieve_all(sids),
        )

        close_loader = DataFrameLoader(USEquityPricing.close, cls.raw_data)

        cls.engine = SimplePipelineEngine(
            {USEquityPricing.close: close_loader}.__getitem__,
            cls.dates,
            cls.asset_finder,
        )

    def test_slice(self):
        """
        Test that slices can be created by indexing into a term, and that they
        have the correct shape when used as inputs.
        """
        my_asset_column = 0
        start_date_index = 5
        end_date_index = 9

        my_asset = self.asset_finder.retrieve_asset(self.sids[0])
        my_asset_only = (AssetID().eq(my_asset_column + 1))

        returns = Returns(window_length=2)
        returns_slice = returns[my_asset]
        self.assertIsInstance(returns_slice, FactorSlice)

        class UsesSlicedInput(CustomFactor):
            window_length = 3
            inputs = [returns, returns_slice]

            def compute(self, today, assets, out, returns, returns_slice):
                # Make sure that our slice is the correct shape (i.e. has only
                # one column) and that it has the same values as the original
                # returns factor from which it is derived.
                assert returns_slice.shape == (self.window_length, 1)
                check_arrays(returns_slice[:, 0], returns[:, my_asset_column])

        columns = {
            'uses_sliced_input': UsesSlicedInput(),
            'uses_sliced_input_masked': UsesSlicedInput(mask=my_asset_only),
        }

        # Assertions about the expected slice data are made in the `compute`
        # function of our custom factor above.
        self.engine.run_pipeline(
            Pipeline(columns=columns),
            self.dates[start_date_index],
            self.dates[end_date_index],
        )

    def test_adding_slice_column(self):
        """
        Test that slices cannot be added as a pipeline column.
        """
        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        with self.assertRaises(UnsupportedPipelineColumn):
            Pipeline(columns={'open_slice': OpenPrice()[my_asset]})

        pipe = Pipeline(columns={})
        with self.assertRaises(UnsupportedPipelineColumn):
            pipe.add(OpenPrice()[my_asset], 'open_slice')

    def test_non_window_safe_slice(self):
        """
        Test that slices of non window safe terms are also non window safe.
        """
        start_date_index = 5
        end_date_index = 9

        my_asset = self.asset_finder.retrieve_asset(self.sids[0])

        # SimpleMovingAverage is not window safe.
        sma = SimpleMovingAverage(
            inputs=[USEquityPricing.close], window_length=3,
        )
        sma_slice = sma[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = 3
            inputs = [sma_slice]

            def compute(self, today, assets, out, sma_slice):
                pass

        with self.assertRaises(NonWindowSafeInput):
            self.engine.run_pipeline(
                Pipeline(columns={'uses_sliced_input': UsesSlicedInput()}),
                self.dates[start_date_index],
                self.dates[end_date_index],
            )

    @parameter_space(returns_length=[2, 3], correlation_length=[3, 4])
    def test_factor_correlation_methods(self,
                                        returns_length,
                                        correlation_length):
        """
        Ensure that `Factor.rolling_pearsonr` and `Factor.rolling_spearmanr`
        are consistent with the built-in factors `RollingPearsonOfReturns` and
        `RollingSpearmanOfReturns`.
        """
        my_asset_column = 0
        start_date_index = 14
        end_date_index = 18

        my_asset = self.asset_finder.retrieve_asset(self.sids[my_asset_column])

        returns = Returns(window_length=returns_length)
        returns_slice = returns[my_asset]

        pearson = returns.rolling_pearsonr(
            target_slice=returns_slice, correlation_length=correlation_length,
        )
        spearman = returns.rolling_spearmanr(
            target_slice=returns_slice, correlation_length=correlation_length,
        )
        expected_pearson = RollingPearsonOfReturns(
            target=my_asset,
            returns_length=returns_length,
            correlation_length=correlation_length,
        )
        expected_spearman = RollingSpearmanOfReturns(
            target=my_asset,
            returns_length=returns_length,
            correlation_length=correlation_length,
        )

        columns = {
            'pearson': pearson,
            'spearman': spearman,
            'expected_pearson': expected_pearson,
            'expected_spearman': expected_spearman,
        }

        results = self.engine.run_pipeline(
            Pipeline(columns=columns),
            self.dates[start_date_index],
            self.dates[end_date_index],
        )
        pearson_results = results['pearson'].unstack()
        spearman_results = results['spearman'].unstack()
        expected_pearson_results = results['expected_pearson'].unstack()
        expected_spearman_results = results['expected_spearman'].unstack()

        assert_frame_equal(pearson_results, expected_pearson_results)
        assert_frame_equal(spearman_results, expected_spearman_results)

    @parameter_space(returns_length=[2, 3], regression_length=[3, 4])
    def test_factor_regression_method(self, returns_length, regression_length):
        """
        Ensure that `Factor.rolling_linear_regression` is consistent with the
        built-in factor `RollingLinearRegressionOfReturns`.
        """
        my_asset_column = 0
        start_date_index = 14
        end_date_index = 18

        my_asset = self.asset_finder.retrieve_asset(self.sids[my_asset_column])

        returns = Returns(window_length=returns_length)
        returns_slice = returns[my_asset]

        regression = returns.rolling_linear_regression(
            target_slice=returns_slice, regression_length=regression_length,
        )
        expected_regression = RollingLinearRegressionOfReturns(
            target=my_asset,
            returns_length=returns_length,
            regression_length=regression_length,
        )

        columns = {
            'regression': regression,
            'expected_regression': expected_regression,
        }

        results = self.engine.run_pipeline(
            Pipeline(columns=columns),
            self.dates[start_date_index],
            self.dates[end_date_index],
        )
        regression_results = results['regression'].unstack()
        expected_regression_results = results['expected_regression'].unstack()

        assert_frame_equal(regression_results, expected_regression_results)

    def test_window_safety_of_slices(self):
        """
        Test that slices correctly inherit the `window_safe` property of the
        term from which they are derived.
        """
        my_asset_column = 0
        my_asset = self.asset_finder.retrieve_asset(self.sids[my_asset_column])

        # These numbers are arbitrary for the purpose of this test.
        returns_length = 2
        correlation_length = 10

        class MyFactor(CustomFactor):
            window_length = 1
            inputs = [USEquityPricing.close]

            def compute(self, today, assets, out, close):
                out[:] = close

        my_factor = MyFactor()
        my_factor_slice = my_factor[my_asset]
        returns = Returns(window_length=returns_length)
        returns_slice = returns[my_asset]

        with self.assertRaises(NonWindowSafeInput):
            my_factor.rolling_pearsonr(
                target_slice=returns_slice,
                correlation_length=correlation_length,
            )

        with self.assertRaises(NonWindowSafeInput):
            returns.rolling_pearsonr(
                target_slice=my_factor_slice,
                correlation_length=correlation_length,
            )

    def test_single_column_output(self):
        """
        Tests for custom factors that compute a 1D out.
        """
        start_date_index = 5
        end_date_index = 9
        alternating_mask = (AssetIDPlusDay() % 2).eq(0)

        class SingleColumnOutput(CustomFactor):
            window_length = 3
            inputs = [USEquityPricing.close]
            window_safe = True
            ndim = 1

            def compute(self, today, assets, out, close):
                # Because we specified ndim as 1, `out` should be a singleton
                # array but `close` should be a regular sized input.
                assert out.shape == (1,)
                assert close.shape == (3, 1)
                out[:] = close.mean()

        class Demean(CustomFactor):
            window_length = 1
            inputs = [USEquityPricing.close, SingleColumnOutput()]

            def compute(self, today, assets, out, close, avg_close):
                # Make sure that `avg_close` has the correct shape. That is, it
                # should always have one column regardless of any mask passed
                # to `Demean`.
                assert avg_close.shape == (1, 1)

        columns = {
            'demean': Demean(),
            'masked_demean': Demean(mask=alternating_mask),
        }

        # Assertions about the expected slice data are made in the `compute`
        # function of our custom factors above.
        self.engine.run_pipeline(
            Pipeline(columns=columns),
            self.dates[start_date_index],
            self.dates[end_date_index],
        )
