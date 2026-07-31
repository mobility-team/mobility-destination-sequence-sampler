use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::SamplerError;

#[derive(Clone, Copy, Debug)]
pub struct OdCostRow {
    pub utility_profile_id: u32,
    pub origin: u32,
    pub destination: u32,
    pub cost: f64,
    pub time: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DestinationInputRow {
    pub activity_id: u32,
    pub destination: u32,
    pub opportunity_capacity: f64,
    pub country_value_coefficient: f64,
    pub saturation_utility: f64,
    pub shadow_price: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub layer: u32,
    pub activity_id: u32,
    pub anchor_id: Option<u32>,
    pub fixed_destination: Option<u32>,
    pub departure_time: f64,
    pub next_departure_time: f64,
    pub duration_per_person: f64,
    pub value_of_time: f64,
    pub mean_duration_per_person: f64,
    pub min_activity_time: f64,
    pub arrival_time: Option<f64>,
    pub arrival_time_rigidity: Option<f64>,
    pub departure_time_rigidity: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub context_id: u64,
    pub utility_profile_id: u32,
    pub initial_zone: u32,
    pub steps: Vec<Step>,
}

fn column_as_vec<'py, T>(df: &Bound<'py, PyAny>, name: &str) -> Result<Vec<T>, SamplerError>
where
    T: FromPyObject<'py>,
{
    let series = df.call_method1("get_column", (name,))?;
    let values = series.call_method0("to_list")?;
    Ok(values.extract()?)
}

fn check_len(name: &str, expected: usize, actual: usize) -> Result<(), SamplerError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SamplerError::InvalidInput(format!(
            "column '{name}' has length {actual}, expected {expected}"
        )))
    }
}

pub fn parse_od_costs(df: &Bound<'_, PyAny>) -> Result<Vec<OdCostRow>, SamplerError> {
    let columns: Vec<String> = df.getattr("columns")?.extract()?;
    let origin: Vec<u32> = column_as_vec(df, "origin")?;
    let destination: Vec<u32> = column_as_vec(df, "destination")?;
    let cost: Vec<f64> = column_as_vec(df, "cost")?;
    let time: Vec<f64> = column_as_vec(df, "time")?;
    let utility_profile_id = if columns.iter().any(|column| column == "utility_profile_id") {
        column_as_vec(df, "utility_profile_id")?
    } else {
        vec![0; origin.len()]
    };
    check_len("destination", origin.len(), destination.len())?;
    check_len("cost", origin.len(), cost.len())?;
    check_len("time", origin.len(), time.len())?;
    check_len("utility_profile_id", origin.len(), utility_profile_id.len())?;

    Ok(utility_profile_id
        .into_iter()
        .zip(origin)
        .zip(destination)
        .zip(cost)
        .zip(time)
        .map(
            |((((utility_profile_id, origin), destination), cost), time)| OdCostRow {
                utility_profile_id,
                origin,
                destination,
                cost,
                time,
            },
        )
        .collect())
}

pub fn parse_destination_inputs(
    df: &Bound<'_, PyAny>,
) -> Result<Vec<DestinationInputRow>, SamplerError> {
    let activity_id: Vec<u32> = column_as_vec(df, "activity_id")?;
    let destination: Vec<u32> = column_as_vec(df, "destination")?;
    let opportunity_capacity: Vec<f64> = column_as_vec(df, "opportunity_capacity")?;
    let country_value_coefficient: Vec<f64> = column_as_vec(df, "country_value_coefficient")?;
    let saturation_utility: Vec<f64> = column_as_vec(df, "saturation_utility")?;
    let shadow_price: Vec<f64> = column_as_vec(df, "shadow_price")?;
    let len = activity_id.len();
    check_len("destination", len, destination.len())?;
    check_len("opportunity_capacity", len, opportunity_capacity.len())?;
    check_len(
        "country_value_coefficient",
        len,
        country_value_coefficient.len(),
    )?;
    check_len("saturation_utility", len, saturation_utility.len())?;
    check_len("shadow_price", len, shadow_price.len())?;

    Ok(activity_id
        .into_iter()
        .zip(destination)
        .zip(opportunity_capacity)
        .zip(country_value_coefficient)
        .zip(saturation_utility)
        .zip(shadow_price)
        .map(
            |(
                (
                    (((activity_id, destination), opportunity_capacity), country_value_coefficient),
                    saturation_utility,
                ),
                shadow_price,
            )| DestinationInputRow {
                activity_id,
                destination,
                opportunity_capacity,
                country_value_coefficient,
                saturation_utility,
                shadow_price,
            },
        )
        .collect())
}

pub fn parse_reference_contexts(
    steps_df: &Bound<'_, PyAny>,
    initial_locations_df: &Bound<'_, PyAny>,
) -> Result<Vec<Context>, SamplerError> {
    let context_id: Vec<u64> = column_as_vec(steps_df, "context_id")?;
    let layer: Vec<u32> = column_as_vec(steps_df, "layer")?;
    let activity_id: Vec<u32> = column_as_vec(steps_df, "activity_id")?;
    let anchor_id: Vec<Option<u32>> = column_as_vec(steps_df, "anchor_id")?;
    let fixed_destination: Vec<Option<u32>> = column_as_vec(steps_df, "fixed_destination")?;
    let departure_time: Vec<f64> = column_as_vec(steps_df, "departure_time")?;
    let next_departure_time: Vec<f64> = column_as_vec(steps_df, "next_departure_time")?;
    let duration_per_person: Vec<f64> = column_as_vec(steps_df, "duration_per_person")?;
    let value_of_time: Vec<f64> = column_as_vec(steps_df, "value_of_time")?;
    let mean_duration_per_person: Vec<f64> = column_as_vec(steps_df, "mean_duration_per_person")?;
    let min_activity_time: Vec<f64> = column_as_vec(steps_df, "min_activity_time")?;
    let arrival_time: Vec<f64> = column_as_vec(steps_df, "arrival_time")?;
    let arrival_time_rigidity: Vec<f64> = column_as_vec(steps_df, "arrival_time_rigidity")?;
    let departure_time_rigidity: Vec<f64> = column_as_vec(steps_df, "departure_time_rigidity")?;
    let len = context_id.len();
    for (name, actual) in [
        ("layer", layer.len()),
        ("activity_id", activity_id.len()),
        ("anchor_id", anchor_id.len()),
        ("fixed_destination", fixed_destination.len()),
        ("departure_time", departure_time.len()),
        ("next_departure_time", next_departure_time.len()),
        ("duration_per_person", duration_per_person.len()),
        ("value_of_time", value_of_time.len()),
        ("mean_duration_per_person", mean_duration_per_person.len()),
        ("min_activity_time", min_activity_time.len()),
        ("arrival_time", arrival_time.len()),
        ("arrival_time_rigidity", arrival_time_rigidity.len()),
        ("departure_time_rigidity", departure_time_rigidity.len()),
    ] {
        check_len(name, len, actual)?;
    }

    let mut steps_by_context: BTreeMap<u64, Vec<Step>> = BTreeMap::new();
    for index in 0..len {
        let step = Step {
            layer: layer[index],
            activity_id: activity_id[index],
            anchor_id: anchor_id[index],
            fixed_destination: fixed_destination[index],
            departure_time: departure_time[index],
            next_departure_time: next_departure_time[index],
            duration_per_person: duration_per_person[index],
            value_of_time: value_of_time[index],
            mean_duration_per_person: mean_duration_per_person[index],
            min_activity_time: min_activity_time[index],
            arrival_time: Some(arrival_time[index]),
            arrival_time_rigidity: Some(arrival_time_rigidity[index]),
            departure_time_rigidity: Some(departure_time_rigidity[index]),
        };
        if !step.departure_time.is_finite()
            || !step.next_departure_time.is_finite()
            || !step.duration_per_person.is_finite()
            || !step.value_of_time.is_finite()
            || !step.mean_duration_per_person.is_finite()
            || !step.min_activity_time.is_finite()
            || !arrival_time[index].is_finite()
            || !arrival_time_rigidity[index].is_finite()
            || !departure_time_rigidity[index].is_finite()
            || step.min_activity_time <= 0.0
            || !(0.0..=1.0).contains(&arrival_time_rigidity[index])
            || !(0.0..=1.0).contains(&departure_time_rigidity[index])
        {
            return Err(SamplerError::InvalidInput(
                "step values must be finite, min_activity_time must be positive, and timing rigidities must be between zero and one"
                    .to_string(),
            ));
        }
        steps_by_context
            .entry(context_id[index])
            .or_default()
            .push(step);
    }

    let initial_context_id: Vec<u64> = column_as_vec(initial_locations_df, "context_id")?;
    let initial_zone: Vec<u32> = column_as_vec(initial_locations_df, "initial_zone")?;
    let initial_columns: Vec<String> = initial_locations_df.getattr("columns")?.extract()?;
    let utility_profile_id = if initial_columns
        .iter()
        .any(|column| column == "utility_profile_id")
    {
        column_as_vec(initial_locations_df, "utility_profile_id")?
    } else {
        vec![0; initial_context_id.len()]
    };
    check_len("initial_zone", initial_context_id.len(), initial_zone.len())?;
    check_len(
        "utility_profile_id",
        initial_context_id.len(),
        utility_profile_id.len(),
    )?;
    let mut initial_by_context = BTreeMap::new();
    for ((context_id, initial_zone), utility_profile_id) in initial_context_id
        .into_iter()
        .zip(initial_zone)
        .zip(utility_profile_id)
    {
        if initial_by_context
            .insert(context_id, (initial_zone, utility_profile_id))
            .is_some()
        {
            return Err(SamplerError::InvalidInput(format!(
                "initial_locations contains duplicate context_id={context_id}"
            )));
        }
    }

    let mut contexts = Vec::with_capacity(steps_by_context.len());
    for (context_id, mut steps) in steps_by_context {
        let (initial_zone, utility_profile_id) =
            initial_by_context.remove(&context_id).ok_or_else(|| {
                SamplerError::InvalidInput(format!(
                    "initial_locations is missing context_id={context_id}"
                ))
            })?;
        steps.sort_unstable_by_key(|step| step.layer);
        if steps
            .iter()
            .enumerate()
            .any(|(index, step)| step.layer as usize != index)
        {
            return Err(SamplerError::InvalidInput(format!(
                "context {context_id} layers must be consecutive and start at zero"
            )));
        }
        contexts.push(Context {
            context_id,
            utility_profile_id,
            initial_zone,
            steps,
        });
    }
    if contexts.is_empty() {
        return Err(SamplerError::InvalidInput(
            "steps must contain at least one context".to_string(),
        ));
    }
    if !initial_by_context.is_empty() {
        return Err(SamplerError::InvalidInput(
            "initial_locations contains context ids that are absent from steps".to_string(),
        ));
    }
    Ok(contexts)
}
