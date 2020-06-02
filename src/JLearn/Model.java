package JLearn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintStream;

public class Model {
	Layer[] layers;
	private int layer_ctr = 0;
	
	public Model(Layer[] layers) {
		this.layers = layers;
	}
	
	public Model(int size) {
		layers = new Layer[size];
	}
	
	public void add(int size, int[] weights_init, int[] bias_init) {
		layers[layer_ctr]= new Layer(size, layers[layers.length - 1].size, weights_init, bias_init);
		layer_ctr++;
	}
	
	public void add(Layer layer) {
		layers[layer_ctr] = layer;
		layer_ctr++;
	}
	
	//Model predicts based off the flow of prediction values from each layer
	public double[] predict(double[] input) {
		for (int i = 0; i < layers.length; i++) {
			input = layers[i].predict(input);
		}
		return input;
	}
	
	/*  Adds to "delta" arrays in each layer based off loss function. 
	 *  Currently only "sigmoid" normalization is supported.
	 *  Used for tuning the model during training.
	 */
	private void propogate(double[] input, double[] output, double[] actual) {
		for (int i = 0; i < output.length; i++) {
			output[i] = costPrime(output[i], actual[i]) * Layer.sigmoidPrime(layers[layers.length - 1].end[i]); 
		}
		for (int i = layers.length - 1; i >= 0; --i) {
			if (i == 0) {
				layers[i].propogate(output);
			} else {
				output = layers[i].propogate(output);
			}
		}
	}
	
	//Optimized loss adjustment for tuning network
	private void adjust(double l_rate) {
		for (int i = 0 ; i < layers.length; i++) {
			layers[i].adjust(l_rate);
		}
	}
	
	/* @param epochs: iterations during training
	 * @param batchSize: number of prediction iterations before applying adjustment
	 * @param x_train: training input data
	 * @param y_train: training output data
	 * @param l_rate: multiplier for weight & bias adjustment function
	 */
	public void train(int epochs, int batchSize, double[][] x_train, double[][] y_train, double l_rate) {
		final int MAX_BATCH = x_train.length / batchSize;
		for (int i = 0; i < epochs; i++) {
			double loss = 0;
			System.out.print("Epoch: " + (i + 1) + ", ");
			long start = System.currentTimeMillis();
			for (int j = 0; j < MAX_BATCH; j++) {
				for (int k = 0; k < batchSize; k++) {
					final int pos = j * batchSize + k;
					double[] temp = predict(x_train[pos]);
					loss += cost(temp, y_train[pos]);
					propogate(
						x_train[pos], 
						temp,
						y_train[pos]
					);
				}
				adjust(l_rate);
			}
			long end = System.currentTimeMillis();
			
			System.out.println("Duration: [" + (Double.valueOf(end - start) / 1000) + " seconds], Loss: " + loss);
		}
	}

	private static double costPrime(double x, double y) {
		return 2 * (x - y);
	}
	
	private static double cost(double x, double y) {
		return Math.pow(x - y, 2);
	}
	
	private static double cost(double[] x, double[] y) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += cost(x[i], y[i]);
		}
		return sum;
	}
	
	//Saves finished model to file
	public static void save(Model model, String path) {
		try {
			PrintStream writer = new PrintStream(path);	
			StringBuffer str = new StringBuffer();
			for (int i = 0; i < model.layers.length; i++) {
				str.append(model.layers[i].dimsToString());
				str.append(" ");
			}
			writer.println(str.toString());
			for (int i = 0; i < model.layers.length; i++) {
				String[] str_weights = model.layers[i].weightsToString();
				for (int j = 0; j < model.layers[i].weights.length; j++) {
					writer.println(str_weights[j]);
				}
				writer.println(model.layers[i].biasToString());
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	//Loads model from file
	public static Model load(String path) {
		try {
			Layer[] layers;
			BufferedReader br = new BufferedReader(new FileReader(new File(path)));
			{
				String[] str_temp = br.readLine().split(" ");
				layers = new Layer[str_temp.length];
				{
					for (int i = 0; i < layers.length; i++) {
						String[] temp = str_temp[i].split(",");
						if (i == 0) {
							layers[i] = new Flattened(Integer.parseInt(temp[0]), Integer.parseInt(temp[1]));
						} else if (i == layers.length - 1) {
							layers[i] = new Classify(Integer.parseInt(temp[0]), Integer.parseInt(temp[1]));
						} else {
							layers[i] = new Layer(Integer.parseInt(temp[0]), Integer.parseInt(temp[1]));
						}	
					}
				}
			}
			for (int i = 0; i < layers.length; i++) {
				for (int j = 0; j < layers[i].weights.length; j++) {
					layers[i].weights[j] = parse(br.readLine());
				}
				layers[i].bias = parse(br.readLine());
			}
			br.close();
			return new Model(layers);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	private static double[] parse(String temp) {
		String[] str_temp = temp.split(",");
		double[] arr = new double[str_temp.length];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = Double.parseDouble(str_temp[i]);
		}
		return arr;
	}
}
