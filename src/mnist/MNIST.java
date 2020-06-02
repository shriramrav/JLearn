package mnist;

import static java.lang.String.format;
import java.io.ByteArrayOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

import JLearn.Matrix;

public class MNIST {
	public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
	public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;
	public static final int IMG_OPT = 255;
	
	public static double[][][] loadData() {
		String[] paths = new String[] {
			("src/mnist/train_images.idx3-ubyte"),
			( "src/mnist/train_labels.idx1-ubyte")
		};
		double[][] y_train;
		List<double[]> flattened_images = new ArrayList<>();
		{
			List<double[][]> temp_images = MNIST.getImages(paths[0]);
			int[] temp_labels = MNIST.getLabels(paths[1]);
			y_train = new double[temp_images.size()][];
			for (int i = 0; i < temp_images.size(); i++) {
				flattened_images.add(Matrix.flatten(temp_images.get(i)));
				y_train[i] = arrayfromint(temp_labels[i]);
			}
		}
		double[][] x_train = new double[flattened_images.size()][];
		flattened_images.toArray(x_train);
		return new double[][][] { x_train, y_train };
	}

	public static void render(double[] image) {
		new Window().render(image);
	}
	
	public static int[] getLabels(String infile) {
		ByteBuffer bb = loadFileToByteBuffer(infile);
		assertMagicNumber(LABEL_FILE_MAGIC_NUMBER, bb.getInt());
		int numLabels = bb.getInt();
		int[] labels = new int[numLabels];
		for (int i = 0; i < numLabels; ++i) {
			labels[i] = bb.get() & 0xFF;	
		}
		return labels;
	}

	public static List<double[][]> getImages(String infile) {
		ByteBuffer bb = loadFileToByteBuffer(infile);
		assertMagicNumber(IMAGE_FILE_MAGIC_NUMBER, bb.getInt());
		int numImages = bb.getInt();
		int numRows = bb.getInt();
		int numColumns = bb.getInt();
		List<double[][]> images = new ArrayList<>();
		for (int i = 0; i < numImages; i++) {
			images.add(readImage(numRows, numColumns, bb));	
		}
		return images;
	}

	private static double[][] readImage(int numRows, int numCols, ByteBuffer bb) {
		double[][] image = new double[numRows][];
		for (int row = 0; row < numRows; row++) {
			image[row] = readRow(numCols, bb);	
		}
		return image;
	}

	private static double[] readRow(int numCols, ByteBuffer bb) {
		double[] row = new double[numCols];
		for (int col = 0; col < numCols; ++col) {
			row[col] = Double.valueOf(bb.get() & 0xFF) / IMG_OPT;
		}
		return row;
	}

	private static void assertMagicNumber(int expectedMagicNumber, int magicNumber) {
		if (expectedMagicNumber != magicNumber) {
			switch (expectedMagicNumber) {
			case LABEL_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not a label file.");
			case IMAGE_FILE_MAGIC_NUMBER:
				throw new RuntimeException("This is not an image file.");
			default:
				throw new RuntimeException(
						format("Expected magic number %d, found %d", expectedMagicNumber, magicNumber));
			}
		}
	}

	private static ByteBuffer loadFileToByteBuffer(String infile) {
		return ByteBuffer.wrap(loadFile(infile));
	}
	
	private static double[] arrayfromint(int n) {
		double[] array = new double[10];
		for (int i = 0; i < 10; i++) {
			array[i] = 0;
			if (i == n) {
				array[i] = 1.0;
			}
		}
		return array;
	}

	private static byte[] loadFile(String infile) {
		try {
			RandomAccessFile f = new RandomAccessFile(infile, "r");
			FileChannel chan = f.getChannel();
			long fileSize = chan.size();
			ByteBuffer bb = ByteBuffer.allocate((int) fileSize);
			chan.read(bb);
			bb.flip();
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			for (int i = 0; i < fileSize; i++) {
				baos.write(bb.get());	
			}
			chan.close();
			f.close();
			return baos.toByteArray();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}