package JLearn;
public class Layer {
	protected int size;
	public double[][] weights;
	public double[][] delta_weights;
	protected double[] bias;
	public double[] delta_bias;
	public double[] z;
	public double[] end;

	public Layer( int input_size, int output_size, int[] weights_init, int[] bias_init) {
		this.size = output_size;
		weights = init(weights_init, new double[input_size][size]);
		bias = init(bias_init, new double[size]);
		delta_weights = zero(new double[input_size][size]);
		delta_bias = zero(new double[size]);
	}
	
	public Layer( int input_size, int output_size) {
		this.size = output_size;
		weights = new double[input_size][size];
		bias = new double[size];
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
		return Matrix.prod(normalize(input), weights, bias);
	}
	
	//Performs Back-propogation based off derivatives of the next layer
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
	
	//Optimized weight and bias adjustment for training model
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
	
	protected static double[] normalize(double[] input) {
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
	
	public String dimsToString() {
		return Integer.toString(weights.length) + "," + Integer.toString(weights[0].length);
	}
	
	public String[] weightsToString() {
		String[] arr = new String[weights.length];
		for (int i = 0; i < weights.length; i++) {
			StringBuffer str = new StringBuffer();
			for (int j = 0; j < weights[0].length; j++) {
				str.append(Double.toString(weights[i][j]));
				if (j != weights[0].length - 1) {
					str.append(",");	
				}
			}
			arr[i] = str.toString();
		}
		return arr;
	}
	
	public String biasToString() {
		StringBuffer str = new StringBuffer();
		for (int i = 0; i < bias.length; i++) {
			str.append(Double.toString(bias[i]));
			if (i != bias.length - 1) {
				str.append(",");	
			}
		}
		return str.toString();
	}	
}
