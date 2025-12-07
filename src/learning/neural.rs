use rand::Rng;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut rng = rand::thread_rng();

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            
            let mut weights = Vec::new();
            for _ in 0..output_size {
                let mut row = Vec::new();
                for _ in 0..input_size {
                    // Xavier initialization
                    let weight = rng.gen_range(-1.0..1.0) * (2.0 / (input_size + output_size) as f32).sqrt();
                    row.push(weight);
                }
                weights.push(row);
            }

            let mut biases = Vec::new();
            for _ in 0..output_size {
                biases.push(rng.gen_range(-0.1..0.1));
            }

            layers.push(Layer { weights, biases });
        }

        Self { layers }
    }

    pub fn from_weights(layer_sizes: &[usize], weights: &[f32]) -> Self {
        let mut layers = Vec::new();
        let mut weight_idx = 0;

        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            
            let mut layer_weights = Vec::new();
            for _ in 0..output_size {
                let mut row = Vec::new();
                for _ in 0..input_size {
                    row.push(weights[weight_idx]);
                    weight_idx += 1;
                }
                layer_weights.push(row);
            }

            let mut biases = Vec::new();
            for _ in 0..output_size {
                biases.push(weights[weight_idx]);
                weight_idx += 1;
            }

            layers.push(Layer {
                weights: layer_weights,
                biases,
            });
        }

        Self { layers }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut activations = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut next_activations = Vec::new();

            for (j, weights) in layer.weights.iter().enumerate() {
                let mut sum = layer.biases[j];
                for (k, &weight) in weights.iter().enumerate() {
                    sum += weight * activations[k];
                }
                
                // ReLU для скрытых слоев, sigmoid для выходного
                let activation = if i == self.layers.len() - 1 {
                    sigmoid(sum) // Выходной слой: активации мышц от 0 до 1
                } else {
                    relu(sum) // Скрытые слои
                };
                
                next_activations.push(activation);
            }

            activations = next_activations;
        }

        activations
    }

    pub fn get_weights(&self) -> Vec<f32> {
        let mut weights = Vec::new();
        for layer in &self.layers {
            for row in &layer.weights {
                weights.extend_from_slice(row);
            }
            weights.extend_from_slice(&layer.biases);
        }
        weights
    }

    pub fn set_weights(&mut self, weights: &[f32]) {
        let mut weight_idx = 0;
        for layer in &mut self.layers {
            for row in &mut layer.weights {
                for weight in row.iter_mut() {
                    *weight = weights[weight_idx];
                    weight_idx += 1;
                }
            }
            for bias in &mut layer.biases {
                *bias = weights[weight_idx];
                weight_idx += 1;
            }
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.get_weights().len()
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = NeuralNetwork::new(&[3, 4, 2]);
        assert_eq!(network.layers.len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let network = NeuralNetwork::new(&[2, 3, 1]);
        let output = network.forward(&[1.0, 0.5]);
        assert_eq!(output.len(), 1);
        assert!(output[0] >= 0.0 && output[0] <= 1.0);
    }
}

