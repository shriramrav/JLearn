package mnist;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import javax.swing.JFrame;

@SuppressWarnings("serial")
public class Window extends JFrame {
	public Window() {
		setSize(784, 784);
	}

	public void render(double[] image) {
		addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
		add(new Render(image));
		setVisible(true);
	}
}
