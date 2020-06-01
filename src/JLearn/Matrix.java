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

//	public static double[][] reduce(double[][] matrix, int factor) {
//		double[][] new_matrix = new double[matrix.length / factor][matrix[0].length / factor];
//		int k = 0, l = 0;
//	
//		for (int i = 0; i < matrix.length; i += factor) {
//			for (int j = 0; j < matrix[0].length; j += factor) {
//				new_matrix[k][l] = 
//				l++;
//			}
//			k++;
//		}
//	}
//	
	
	public static double[][] reduce(double[][] matrix) {
		double[][] new_matrix = new double[14][14];
		int k = 0, l = 0;
	
		for (int i = 0; i < matrix.length; i += 2) {
			for (int j = 0; j < matrix[0].length; j += 2) {
				new_matrix[k][l] = ((matrix[i][j]) + (matrix[i][j + 1]) + (matrix[i + 1][j]) + (matrix[i + 1][j + 1])) / 4;
				l++;
			}
			k++;
		}
		return new_matrix;
	}
	
}
