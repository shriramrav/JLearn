package JLearn;

public class Matrix {
	public static double[] prod(double[] vector, double[][] matrix, double[] bias) {
		double[] sum = new double[matrix[0].length];
		for (int i = 0; i < sum.length; i++) {
			sum[i] = Matrix.dot(vector, matrix, i) + bias[i];
		}
		return sum;
	}

	public static double dot(double[] vector, double[][] matrix, int pos) {
		double sum = 0;
		for (int i = 0; i < vector.length; i++) {
			sum += vector[i] * matrix[i][pos];
		}
		return sum;
	}
	
	public static double[][] reduce(double[][] matrix) {
		double[][] new_matrix = new double[matrix.length / 2][matrix[0].length / 2];
		int k = 0, l = 0;
	
		for (int i = 0; i < matrix.length; i += 2) {
			l = 0;
			for (int j = 0; j < matrix[0].length; j += 2) {
				new_matrix[k][l] = ((matrix[i][j]) + (matrix[i][j + 1]) + (matrix[i + 1][j]) + (matrix[i + 1][j + 1])) / 4;
				l++;
			}
			k++;
		}
		return new_matrix;
	}
	
	public static double[] flatten(double[][] matrix) {
		double[] new_matrix = new double[matrix.length * matrix[0].length];
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[0].length; j++) {
				new_matrix[i * matrix[0].length + j] = matrix[i][j];
			}
		}
		return new_matrix;
	}
	
	public static double[] softmax(double[] vector) {
		double sum = 0;
		for (int i = 0; i < vector.length; i++) {
			sum += vector[i];
		}
		for (int i = 0; i < vector.length; i++) {
			vector[i] /= sum;
		}
		return vector;
	}
	
	public static int argMax(double[] vector) {
		int pos = 0;
		for (int i = 1; i < vector.length; i++) {
			if (vector[pos] < vector[i]) {
				pos = i;
			}
			
		}
		return pos;
	}
}
