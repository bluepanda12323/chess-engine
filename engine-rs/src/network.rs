struct Layer {
    outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

pub struct Network {
    layers: Vec<Layer>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> u32 {
        let value = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        value
    }

    fn f32s(&mut self, count: usize) -> Vec<f32> {
        let values = self.bytes[self.pos..self.pos + 4 * count]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        self.pos += 4 * count;
        values
    }
}

impl Network {
    pub fn load(path: &str) -> std::io::Result<Network> {
        let bytes = std::fs::read(path)?;
        let mut reader = Reader { bytes: &bytes, pos: 0 };

        let num_layers = reader.u32() as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let outputs = reader.u32() as usize;
            let inputs = reader.u32() as usize;
            let weights = reader.f32s(outputs * inputs);
            let bias = reader.f32s(outputs);

            let mut transposed = vec![0.0; inputs * outputs];
            for o in 0..outputs {
                for i in 0..inputs {
                    transposed[i * outputs + o] = weights[o * inputs + i];
                }
            }
            layers.push(Layer { outputs, weights: transposed, bias });
        }

        Ok(Network { layers })
    }

    pub fn evaluate(&self, input: &[f32]) -> f32 {
        let mut current = input.to_vec();
        let last = self.layers.len() - 1;

        for (index, layer) in self.layers.iter().enumerate() {
            let mut next = layer.bias.clone();
            for (k, &x) in current.iter().enumerate() {
                if x != 0.0 {
                    let column = &layer.weights[k * layer.outputs..(k + 1) * layer.outputs];
                    for (out, &w) in next.iter_mut().zip(column) {
                        *out += x * w;
                    }
                }
            }

            if index < last {
                for value in &mut next {
                    *value = value.max(0.0);
                }
            } else {
                for value in &mut next {
                    *value = value.tanh();
                }
            }
            current = next;
        }

        current[0]
    }
}
