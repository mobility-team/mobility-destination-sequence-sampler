//! Shared search parameters and fixed-destination lookup.

use crate::model::DestinationValue;

#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub logit_scale: f64,
    pub update_plan_timings: bool,
    pub use_shadow_prices: bool,
    pub seed: u64,
    pub n_draws: u32,
    pub skip_infeasible: bool,
}

#[inline]
pub(crate) fn fixed_destination_value(
    activity_values: Option<&[Option<DestinationValue>]>,
    destination: usize,
) -> DestinationValue {
    activity_values
        .and_then(|values| values[destination])
        .unwrap_or(DestinationValue {
            log_opportunity_capacity: 0.0,
            country_value_coefficient: 1.0,
            saturation_utility: 1.0,
            shadow_price: 0.0,
        })
}
