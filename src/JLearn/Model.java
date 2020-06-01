package JLearn;


public class Model {
	Layer[] layers;
	private int layer_ctr = 0;
	
	Model(int size) {
		layers = new Layer[size];
	}
	
	Model(Layer[] layers) {
		this.layers = layers;
	}
	
	public void add(int size, int[] weights_init, int[] bias_init) {
		layers[layer_ctr]= new Layer(size, layers[layers.length - 1].size, weights_init, bias_init);
		layer_ctr++;
	}
	
	public void add(Layer layer) {
		layers[layer_ctr] = layer;
		layer_ctr++;
	}
	
	public double[] predict(double[] input) {
		for (int i = 0; i < layers.length; i++) {
			input = layers[i].predict(input);
		}
		return input;
	}
	
	private void adjust(double l_rate) {
		for (int i = 0 ; i < layers.length; i++) {
			layers[i].adjust(l_rate);
		}
	}
	
	private void propogate(double[] input, double[] output, double[] actual) {
		for (int i = 0; i < output.length; i++) {
			output[i] = costPrime(output[i], actual[i]) * Layer.sigmoidPrime(layers[layers.length - 1].end[i]); 
		}
		for (int i = layers.length - 1; i >= 0; --i) {
			if (i == 0) {
				layers[i].propogate(output, true);
			} else {
				output = layers[i].propogate(output);
			}
		}
	}
	
	public void train(int epochs, int batchSize, double[][] x_train, double[][] y_train, double l_rate) {
		final int MAX_BATCH = x_train.length / batchSize;
		for (int i = 0; i < epochs; i++) {
			for (int j = 0; j < MAX_BATCH; j++) {
				for (int k = 0; k < batchSize; k++) {
					propogate(
						x_train[(j * batchSize) + k], 
						predict(x_train[(j * batchSize) + k]),
						y_train[(j * batchSize) + k]
					);
				}
				adjust(l_rate);
			}
		}
	}

	private static double costPrime(double x, double y) {
		return 2 * (x - y);
	}
	
	private static double cost(double x, double y) {
		return Math.pow(x - y, 2);
	}
	
	
}
