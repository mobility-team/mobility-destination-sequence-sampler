//! Small scoring primitives shared by active search and retained baselines.

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
pub(crate) fn logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + (-(left - right).abs()).exp().ln_1p()
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

#[inline]
pub(crate) fn alternative_gumbel(
    seed: u64,
    context_id: u64,
    draw_id: u32,
    alternative_index: u64,
) -> f64 {
    let mut value = seed
        ^ context_id.wrapping_mul(0x9E3779B97F4A7C15)
        ^ u64::from(draw_id).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ alternative_index.wrapping_mul(0x94D049BB133111EB);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^= value >> 31;
    let unit = ((value >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
    -(-unit.ln()).ln()
}
