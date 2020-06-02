package JLearn;

public class Flattened extends Layer {

	public Flattened(int input_size, int output_size, int[] weights_init, int[] bias_init) {
		super(input_size, output_size, weights_init, bias_init);
	}
	
	public Flattened(int input_size, int output_size) {
		super(input_size, output_size);
	}

	@Override
	public double[] propogate(double[] derivatives) {
		for (int i = 0; i < super.delta_bias.length; i++) {
			super.delta_bias[i] += derivatives[i];
			for (int j = 0; j < super.delta_weights.length; j++) {
				super.delta_weights[j][i] += derivatives[i] * sigmoid(z[j]);
			}
		}
		return null;
	}
	
	
	//First layer in network, uses raw input from training data
	@Override
	public double[] predict(double[] input) {
		super.z = input;
		return Matrix.prod(input, super.weights, super.bias);
	}
}
