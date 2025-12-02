use std::collections::HashMap;

#[derive(Clone)]
pub struct Genome {
    pub connection_genes: Vec<ConnectionGene>,
    pub node_genes: Vec<NodeGene>,
}

pub struct ConnectionGene {
    pub innovation_number: u64,
    pub from_node: u64,
    pub to_node: u64,
    pub weight: f32,
    pub enabled: bool,
}

pub struct NodeGene {
    pub id: u64,
    pub node_type: NodeType,
    pub activation_function: ActivationFunction,
}

#[derive(Clone, Copy, PartialEq)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
}

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}

pub struct Network {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
}

struct Node {
    id: u64,
    node_type: NodeType,
    activation_function: ActivationFunction,
    value: f32,
}

struct Connection {
    from: usize,
    to: usize,
    weight: f32,
    enabled: bool,
}

static mut INNOVATION_COUNTER: u64 = 0;

pub fn get_next_innovation() -> u64 {
    unsafe {
        INNOVATION_COUNTER += 1;
        INNOVATION_COUNTER
    }
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            connection_genes: vec![],
            node_genes: vec![],
        }
    }
}

impl Network {
    pub fn new(genome: &Genome) -> Self {
        let mut nodes = Vec::new();
        let mut node_map = HashMap::new();
        
        for (i, node_gene) in genome.node_genes.iter().enumerate() {
            node_map.insert(node_gene.id, i);
            nodes.push(Node {
                id: node_gene.id,
                node_type: node_gene.node_type,
                activation_function: node_gene.activation_function,
                value: 0.0,
            });
        }

        let mut connections = Vec::new();
        for conn_gene in &genome.connection_genes {
            if conn_gene.enabled {
                if let (Some(&from_idx), Some(&to_idx)) = 
                    (node_map.get(&conn_gene.from_node), node_map.get(&conn_gene.to_node)) {
                    connections.push(Connection {
                        from: from_idx,
                        to: to_idx,
                        weight: conn_gene.weight,
                        enabled: true,
                    });
                }
            }
        }

        Self { nodes, connections }
    }

    pub fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut input_idx = 0;
        for node in &mut self.nodes {
            if node.node_type == NodeType::Input {
                if input_idx < inputs.len() {
                    node.value = inputs[input_idx];
                    input_idx += 1;
                }
            } else {
                node.value = 0.0;
            }
        }

        for conn in &self.connections {
            if conn.enabled {
                let input_value = self.nodes[conn.from].value;
                self.nodes[conn.to].value += input_value * conn.weight;
            }
        }

        for node in &mut self.nodes {
            if node.node_type != NodeType::Input {
                node.value = Self::activate(node.value, node.activation_function);
            }
        }

        self.nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Output)
            .map(|n| n.value)
            .collect()
    }

    fn activate(x: f32, func: ActivationFunction) -> f32 {
        match func {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Linear => x,
        }
    }
}

