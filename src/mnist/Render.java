package mnist;

import java.awt.Color;
import java.awt.Component;
import java.awt.Graphics;

@SuppressWarnings("serial")
public class Render extends Component {
	double[] image;
	
	//Used to set rendered color
	final int IMG_CST = 0;

	Render(double[] image) {
		this.image = image;
	}

	public void paint(Graphics g) {
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				final int IMG_VALUE = Math.abs(IMG_CST - (int) (image[j * 28 + i] * 255));
				g.setColor(new Color(IMG_VALUE, IMG_VALUE, IMG_VALUE));
				g.fillRect(i * 28, j * 28, 28, 28);
			}
		}
	}
}
