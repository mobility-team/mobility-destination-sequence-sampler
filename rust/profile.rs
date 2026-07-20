use std::time::Duration;

#[derive(Debug, Default)]
pub struct ContextProfile {
    pub total: Duration,
    pub tree_problem_build: Duration,
    pub tree_structure_build: Duration,
    pub tree_backward: Duration,
    pub tree_forward: Duration,
    pub variables: u64,
    pub pair_factors: u64,
    pub pair_transitions: u64,
    pub domain_choices: u64,
    pub outgoing_messages: u64,
    pub message_edges: u64,
}

#[derive(Debug, Default)]
pub struct ProfileReport {
    pub plan_build: Duration,
    pub sampling_wall: Duration,
    pub output_merge: Duration,
    pub context_cpu: Duration,
    pub anchor_cpu: Duration,
    pub tree_cpu: Duration,
    pub tree_problem_build_cpu: Duration,
    pub tree_structure_build_cpu: Duration,
    pub tree_backward_cpu: Duration,
    pub tree_forward_cpu: Duration,
    pub contexts: u64,
    pub anchor_contexts: u64,
    pub tree_contexts: u64,
    pub successful_contexts: u64,
    pub infeasible_contexts: u64,
    pub cyclic_contexts: u64,
    pub input_steps: u64,
    pub output_rows: u64,
    pub variables: u64,
    pub pair_factors: u64,
    pub pair_transitions: u64,
    pub domain_choices: u64,
    pub outgoing_messages: u64,
    pub message_edges: u64,
}

impl ProfileReport {
    pub fn add_context(&mut self, profile: &ContextProfile, is_anchor: bool) {
        self.context_cpu += profile.total;
        if is_anchor {
            self.anchor_cpu += profile.total;
        } else {
            self.tree_cpu += profile.total;
        }
        self.tree_problem_build_cpu += profile.tree_problem_build;
        self.tree_structure_build_cpu += profile.tree_structure_build;
        self.tree_backward_cpu += profile.tree_backward;
        self.tree_forward_cpu += profile.tree_forward;
        self.variables += profile.variables;
        self.pair_factors += profile.pair_factors;
        self.pair_transitions += profile.pair_transitions;
        self.domain_choices += profile.domain_choices;
        self.outgoing_messages += profile.outgoing_messages;
        self.message_edges += profile.message_edges;
    }
}
