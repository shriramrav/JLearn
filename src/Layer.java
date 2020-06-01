public class Layer {
	protected int size;
	public double[][] weights;
	public double[] bias;
	public double[][] delta_weights;
	public double[] delta_bias;
	public double[] z;
	public double[] end;

	Layer( int input_size, int output_size, int[] weights_init, int[] bias_init) {
		this.size = output_size;
		weights = init(weights_init, new double[input_size][size]);
		bias = init(bias_init, new double[size]);
		delta_weights = zero(new double[input_size][size]);
		delta_bias = zero(new double[size]);
	}

	private static double[][] init(int[] bounds, double[][] array) {
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				array[i][j] = Layer.rand(bounds);
			}
		}
		return array;
	}

	private static double[] init(int[] bounds, double[] array) {
		for (int i = 0; i < array.length; i++) {
			array[i] = Layer.rand(bounds);
		}
		return array;
	}
	
	private static double[] zero(double[] array) {
		for (int i = 0; i < array.length; i++) {
			array[i] = 0;
		}
		return array;
	}

	private static double[][] zero(double[][] array) {
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				array[i][j] = 0;
			}
		}
		return array;
	}
	
	public double[] predict(double[] input) {
		z = input;
		return Matrix.prod(apply(input), weights, bias);
	}
	
	public double[] propogate(double[] derivatives) {
		double[] vector = zero(new double[delta_weights.length]);
		for (int i = 0; i < delta_weights[0].length; i++) {
			delta_bias[i] += derivatives[i];
			for (int j = 0; j < delta_weights.length; j++) {
				delta_weights[j][i] += derivatives[i] * sigmoid(z[j]);
				vector[j] += derivatives[i] * weights[j][i];
				if (i == delta_weights[0].length - 1) {
					vector[j] *= sigmoidPrime(z[j]);
				}
			}
		}
		return vector;
	}
	
	public void propogate(double[] derivatives, boolean last) {
		for (int i = 0; i < delta_bias.length; i++) {
//			System.out.println("delta bias: " + delta_bias.length + ", " + derivatives.length);
			delta_bias[i] += derivatives[i];
			for (int j = 0; j < delta_weights.length; j++) {
				delta_weights[j][i] += derivatives[i] * sigmoid(z[j]);
			}
		}
	}
	
	public void adjust(double l_rate) {
		for (int i = 0; i < bias.length; i++) {
			bias[i] -= l_rate * delta_bias[i];
			delta_bias[i]  = 0;
			for (int j = 0; j < weights.length; j++) {
				weights[j][i] -= l_rate * delta_weights[j][i];
				delta_weights[j][i] = 0;
			}
		}
	}
	
	public double[] apply(double[] input) {
		for (int i = 0; i < input.length; i++) {
			input[i] = sigmoid(input[i]);
		}
		return input;
	}
	
	protected static double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	public static double sigmoidPrime(double x) {
		return Layer.sigmoid(x) * (1 - Layer.sigmoid(x));
	}
	
	public static double rand(int[] bounds) {
		return Math.random() * (bounds[1] - bounds[0]) + bounds[0];
	}
}
