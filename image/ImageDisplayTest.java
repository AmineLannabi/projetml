/**
 * JUnit test for ImageDisplay class
 * 
 * @author Théophile Walter
 */

package image;


public class ImageDisplayTest {

	public void ImageDisplay() throws InterruptedException {

		// Creating images
		Image img1 = new Image (IMG1(), 32, 32);
		Image img2 = new Image (IMG2(), 32, 32);
		
		// Creating some displays
		ImageDisplay display1 = new ImageDisplay(img1);
		ImageDisplay display2 = new ImageDisplay(img2);
		
		// Displaying two of them
		display1.setVisible(true);
		display2.setVisible(true);
		
		// 5 seconds
		Thread.sleep(5000);
		
		// Update an image
		display1.setImage(img2);
		
		// 5 seconds
		Thread.sleep(5000);
		
		// Hide both
		display1.setVisible(false);
		display2.setVisible(false);
		
		// 2 seconds
		Thread.sleep(5000);
		
		// ;)
	}
	
	/**
	 * Return a char[] image
	 */
	public static char[] IMG1() {
		
		return new char[] {28, 25, 10, 37, 34, 19, 38, 35, 20, 42, 37, 23, 44, 39, 25, 40, 37, 22, 40, 38, 23, 24, 23, 9, 32, 25, 15, 43, 27, 19, 30, 20, 10, 32, 30, 17, 41, 37, 23, 52, 48, 34,
				67, 63, 50, 50, 46, 32, 44, 35, 25, 43, 35, 25, 38, 29, 20, 38, 30, 20, 41, 34, 23, 47, 39, 25, 62, 50, 33, 87, 71, 50, 60, 48, 27, 49, 42, 24, 63, 56, 41, 59, 51, 40,
				48, 40, 25, 76, 67, 39, 81, 72, 43, 85, 76, 47, 33, 28, 13, 34, 30, 14, 32, 27, 12, 39, 33, 18, 35, 29, 14, 38, 33, 17, 38, 34, 18, 40, 36, 19, 54, 47, 30, 48, 34, 19,
				28, 18, 4, 40, 35, 19, 56, 47, 32, 66, 57, 42, 79, 69, 55, 47, 37, 25, 42, 34, 18, 50, 42, 25, 64, 56, 40, 47, 39, 23, 55, 48, 31, 81, 73, 53, 84, 73, 52, 109, 95, 70,
				92, 80, 58, 59, 51, 32, 69, 61, 42, 79, 72, 52, 71, 62, 40, 95, 82, 55, 96, 82, 56, 85, 72, 45, 39, 32, 15, 40, 33, 17, 57, 50, 33, 46, 41, 23, 44, 38, 21, 40, 34, 17,
				41, 33, 17, 66, 60, 40, 90, 81, 59, 57, 45, 24, 48, 38, 19, 72, 64, 43, 74, 60, 45, 76, 61, 46, 93, 78, 63, 66, 52, 38, 65, 55, 34, 80, 70, 49, 90, 81, 59, 67, 58, 36,
				88, 81, 56, 96, 87, 61, 83, 73, 46, 105, 94, 65, 107, 96, 73, 86, 76, 57, 89, 80, 57, 93, 85, 56, 86, 75, 47, 93, 76, 52, 107, 89, 66, 95, 77, 54, 54, 46, 28, 62, 55, 36,
				84, 77, 58, 62, 58, 39, 70, 65, 46, 55, 48, 30, 78, 68, 51, 85, 75, 54, 99, 90, 64, 84, 74, 48, 95, 85, 59, 90, 79, 55, 96, 79, 61, 103, 85, 68, 93, 76, 59, 95, 78, 60,
				94, 83, 58, 100, 90, 64, 94, 84, 58, 87, 77, 51, 92, 83, 54, 89, 80, 50, 96, 87, 56, 96, 87, 56, 83, 72, 48, 76, 65, 45, 95, 85, 58, 104, 96, 61, 97, 86, 53, 101, 80, 59,
				99, 78, 57, 91, 70, 49, 74, 66, 47, 72, 64, 45, 78, 69, 50, 85, 82, 61, 95, 90, 70, 77, 68, 49, 103, 89, 72, 98, 84, 62, 80, 70, 42, 93, 84, 55, 99, 90, 61, 93, 79, 53,
				101, 83, 62, 105, 87, 67, 84, 66, 45, 84, 67, 45, 86, 75, 48, 87, 77, 49, 80, 71, 42, 87, 78, 49, 80, 69, 38, 91, 81, 48, 93, 84, 53, 74, 66, 36, 81, 71, 47, 77, 66, 45,
				89, 79, 51, 82, 73, 39, 84, 73, 40, 78, 59, 37, 72, 53, 32, 84, 64, 43, 76, 66, 45, 68, 57, 37, 69, 58, 38, 74, 71, 48, 78, 74, 51, 81, 71, 51, 89, 73, 54, 90, 73, 51,
				79, 67, 39, 66, 59, 30, 81, 71, 43, 91, 75, 49, 87, 72, 47, 86, 71, 46, 90, 74, 49, 85, 70, 44, 81, 69, 41, 82, 71, 43, 78, 67, 39, 85, 74, 46, 87, 75, 43, 91, 80, 48,
				83, 75, 46, 80, 73, 46, 87, 78, 54, 91, 79, 55, 98, 86, 61, 74, 63, 35, 59, 47, 20, 71, 55, 35, 72, 56, 35, 94, 78, 58, 77, 66, 44, 74, 63, 41, 73, 62, 40, 70, 68, 43,
				67, 63, 39, 84, 72, 50, 94, 75, 55, 98, 77, 56, 82, 67, 42, 72, 66, 38, 75, 64, 37, 75, 57, 32, 81, 69, 41, 87, 75, 47, 91, 80, 52, 82, 70, 42, 90, 77, 50, 85, 71, 45,
				95, 81, 55, 103, 89, 63, 104, 90, 60, 91, 78, 48, 78, 69, 43, 79, 74, 50, 95, 86, 62, 95, 81, 57, 91, 76, 54, 71, 57, 37, 65, 51, 32, 85, 73, 53, 80, 68, 49, 80, 67, 48,
				79, 68, 46, 76, 64, 43, 85, 74, 52, 77, 75, 50, 83, 79, 54, 81, 68, 47, 94, 73, 54, 122, 98, 78, 94, 79, 55, 82, 75, 49, 93, 81, 56, 86, 68, 44, 101, 93, 62, 91, 83, 52,
				91, 83, 52, 102, 93, 63, 95, 81, 57, 93, 78, 55, 100, 86, 62, 111, 96, 72, 103, 87, 60, 98, 84, 57, 93, 84, 60, 78, 73, 53, 90, 81, 59, 93, 78, 53, 83, 68, 48, 71, 55, 40,
				74, 59, 45, 89, 79, 59, 77, 68, 47, 56, 47, 26, 52, 47, 31, 51, 46, 30, 62, 57, 41, 74, 72, 52, 75, 69, 51, 87, 71, 55, 121, 98, 72, 129, 108, 79, 97, 81, 55, 99, 85, 55,
				115, 95, 64, 128, 105, 76, 121, 107, 75, 83, 69, 36, 99, 85, 50, 95, 82, 46, 87, 71, 42, 92, 74, 50, 95, 72, 51, 103, 79, 55, 101, 80, 58, 101, 85, 60, 86, 73, 47, 74, 64, 47,
				84, 70, 52, 105, 86, 62, 111, 92, 70, 74, 55, 36, 93, 76, 57, 94, 82, 59, 91, 79, 56, 75, 63, 39, 45, 43, 29, 27, 25, 11, 37, 36, 22, 55, 52, 35, 60, 52, 39, 83, 64, 50,
				113, 88, 58, 109, 88, 54, 97, 82, 54, 102, 81, 50, 108, 82, 48, 136, 108, 76, 89, 71, 42, 95, 77, 46, 111, 94, 59, 104, 87, 50, 112, 96, 64, 113, 93, 69, 109, 83, 62, 117, 87, 62,
				104, 79, 59, 111, 92, 68, 77, 62, 34, 74, 60, 43, 96, 79, 63, 97, 77, 53, 101, 81, 57, 83, 63, 39, 106, 87, 63, 88, 74, 51, 80, 67, 43, 88, 75, 51, 67, 61, 42, 41, 35, 17,
				54, 48, 29, 54, 48, 29, 63, 52, 37, 92, 73, 56, 112, 87, 58, 99, 78, 48, 91, 74, 50, 104, 82, 54, 149, 121, 90, 125, 97, 67, 82, 64, 41, 109, 92, 67, 107, 91, 62, 102, 87, 55,
				122, 108, 75, 116, 98, 73, 116, 92, 70, 120, 94, 67, 93, 69, 47, 89, 69, 44, 92, 76, 48, 80, 66, 47, 93, 77, 58, 100, 82, 58, 96, 78, 54, 110, 93, 68, 108, 93, 69, 71, 60, 39,
				69, 58, 37, 96, 85, 64, 79, 68, 45, 82, 70, 47, 91, 80, 57, 93, 83, 61, 69, 54, 36, 88, 69, 49, 123, 99, 71, 96, 75, 50, 94, 75, 56, 105, 82, 57, 136, 106, 77, 104, 75, 47,
				77, 59, 41, 109, 92, 72, 95, 79, 55, 93, 78, 50, 112, 98, 64, 100, 84, 56, 114, 94, 69, 114, 94, 62, 92, 70, 43, 92, 72, 45, 106, 89, 61, 101, 86, 65, 105, 90, 68, 118, 103, 76,
				121, 105, 79, 119, 103, 77, 86, 73, 48, 72, 64, 45, 75, 67, 48, 83, 75, 56, 91, 74, 48, 102, 84, 59, 107, 90, 64, 122, 106, 82, 102, 84, 61, 85, 65, 42, 80, 58, 33, 76, 56, 35,
				83, 62, 47, 112, 86, 64, 110, 79, 50, 85, 55, 27, 69, 51, 35, 95, 77, 59, 95, 78, 57, 92, 76, 50, 96, 82, 47, 91, 76, 47, 94, 77, 50, 104, 87, 53, 110, 89, 58, 111, 90, 60,
				104, 86, 58, 100, 84, 60, 96, 82, 56, 104, 90, 61, 100, 86, 57, 93, 79, 50, 83, 71, 45, 81, 74, 56, 81, 74, 56, 87, 80, 62, 119, 98, 70, 116, 96, 67, 133, 112, 84, 136, 113, 87,
				127, 104, 78, 126, 105, 80, 59, 40, 17, 57, 38, 20, 73, 51, 36, 88, 61, 39, 111, 78, 47, 101, 70, 39, 69, 48, 30, 75, 55, 35, 91, 72, 49, 102, 83, 55, 117, 100, 64, 130, 111, 82,
				143, 126, 97, 154, 138, 102, 150, 129, 93, 116, 93, 62, 114, 94, 66, 104, 87, 60, 94, 78, 50, 98, 83, 53, 101, 86, 55, 94, 79, 48, 92, 79, 52, 89, 80, 61, 84, 75, 56, 82, 73, 54,
				118, 97, 68, 123, 102, 72, 136, 115, 86, 131, 104, 76, 134, 107, 77, 133, 112, 81, 56, 38, 19, 58, 40, 25, 75, 51, 36, 88, 59, 34, 142, 109, 72, 162, 130, 93, 129, 105, 81, 136, 112, 86,
				150, 127, 97, 153, 130, 98, 131, 108, 71, 189, 166, 136, 213, 192, 162, 209, 191, 154, 195, 172, 133, 133, 109, 76, 114, 93, 65, 108, 90, 59, 115, 98, 65, 113, 96, 64, 111, 93, 61, 116, 98, 67,
				99, 83, 54, 98, 85, 62, 83, 71, 48, 79, 66, 44, 88, 68, 40, 100, 79, 51, 110, 89, 61, 126, 96, 67, 125, 96, 62, 122, 100, 67, 66, 51, 32, 60, 42, 28, 97, 72, 53, 104, 74, 44,
				177, 143, 99, 184, 150, 105, 164, 135, 102, 170, 142, 107, 179, 151, 112, 181, 153, 113, 152, 121, 85, 182, 152, 124, 208, 181, 153, 227, 206, 168, 211, 187, 145, 164, 138, 104, 140, 117, 90, 126, 106, 74,
				132, 112, 78, 136, 113, 82, 132, 108, 77, 135, 112, 80, 113, 91, 61, 116, 99, 73, 97, 81, 55, 109, 92, 66, 78, 59, 33, 86, 68, 41, 94, 75, 48, 128, 96, 66, 126, 94, 58, 114, 92, 57,
				95, 80, 61, 53, 36, 22, 102, 76, 54, 129, 100, 65, 198, 164, 116, 201, 165, 114, 189, 155, 114, 195, 162, 118, 205, 172, 126, 208, 175, 127, 181, 144, 109, 173, 138, 110, 190, 160, 131, 207, 183, 145,
				216, 191, 146, 174, 148, 112, 154, 130, 103, 126, 107, 72, 131, 109, 73, 145, 117, 87, 146, 117, 88, 141, 112, 82, 127, 100, 71, 124, 104, 74, 111, 91, 61, 116, 95, 66, 80, 62, 38, 79, 60, 35,
				93, 72, 47, 121, 90, 59, 128, 98, 60, 116, 93, 58, 92, 74, 52, 40, 19, 6, 92, 64, 40, 162, 130, 89, 207, 171, 117, 219, 181, 127, 216, 182, 136, 211, 178, 131, 208, 175, 125, 211, 178, 128,
				201, 164, 122, 179, 142, 113, 165, 139, 113, 139, 116, 85, 175, 152, 115, 152, 127, 98, 126, 102, 77, 122, 102, 65, 142, 116, 78, 150, 117, 85, 157, 124, 93, 153, 121, 91, 134, 104, 74, 132, 105, 74,
				134, 109, 77, 120, 97, 65, 104, 82, 62, 111, 87, 64, 121, 94, 70, 130, 104, 71, 126, 101, 65, 121, 96, 64, 106, 82, 54, 75, 46, 29, 108, 74, 49, 188, 152, 105, 219, 178, 120, 233, 192, 135,
				225, 194, 145, 222, 192, 142, 208, 177, 127, 219, 189, 137, 216, 188, 133, 177, 143, 110, 136, 121, 100, 97, 80, 62, 115, 94, 77, 101, 79, 65, 96, 74, 53, 138, 115, 78, 154, 124, 83, 151, 114, 78,
				157, 122, 88, 152, 120, 87, 142, 110, 79, 139, 106, 76, 140, 112, 81, 116, 94, 61, 103, 84, 64, 119, 96, 74, 126, 100, 76, 135, 111, 78, 122, 98, 64, 116, 91, 61, 121, 95, 68, 123, 94, 74,
				132, 99, 73, 193, 157, 111, 223, 182, 126, 230, 190, 133, 225, 194, 137, 229, 198, 141, 220, 189, 132, 226, 196, 138, 225, 194, 137, 182, 145, 112, 94, 81, 61, 83, 70, 56, 97, 77, 66, 90, 70, 60,
				96, 73, 58, 140, 117, 85, 153, 125, 86, 149, 116, 80, 143, 113, 79, 130, 103, 71, 134, 106, 77, 128, 100, 72, 124, 99, 68, 110, 87, 54, 97, 79, 60, 106, 85, 64, 121, 98, 74, 117, 95, 63,
				112, 91, 58, 122, 100, 71, 125, 102, 77, 109, 83, 62, 116, 88, 61, 178, 145, 102, 218, 179, 128, 222, 183, 129, 227, 194, 134, 234, 200, 140, 219, 186, 125, 226, 193, 132, 219, 184, 125, 192, 152, 116,
				70, 57, 36, 63, 56, 44, 76, 59, 51, 82, 63, 57, 93, 71, 60, 130, 107, 82, 136, 111, 76, 143, 117, 82, 128, 105, 73, 104, 84, 54, 103, 84, 57, 103, 83, 58, 104, 84, 55, 113, 91, 58,
				77, 62, 44, 103, 84, 63, 108, 87, 64, 104, 85, 55, 122, 104, 73, 109, 90, 63, 105, 86, 61, 98, 77, 55, 107, 82, 55, 138, 109, 69, 180, 146, 100, 186, 149, 102, 216, 181, 128, 229, 193, 141,
				211, 175, 122, 216, 180, 126, 212, 174, 116, 199, 155, 118, 54, 40, 21, 16, 13, 5, 49, 34, 30, 67, 47, 45, 86, 65, 59, 119, 96, 77, 113, 92, 61, 117, 98, 65, 104, 89, 57, 87, 75, 46,
				85, 74, 48, 86, 74, 52, 91, 75, 48, 93, 74, 41, 48, 36, 18, 82, 67, 47, 78, 61, 38, 107, 90, 62, 111, 94, 66, 95, 78, 53, 91, 74, 52, 94, 76, 55, 97, 78, 49, 97, 73, 35,
				142, 111, 72, 116, 82, 47, 136, 100, 64, 151, 115, 80, 114, 78, 44, 113, 78, 43, 191, 153, 100, 198, 152, 114, 71, 56, 34, 25, 26, 16, 30, 18, 14, 47, 28, 28, 72, 52, 48, 105, 84, 69,
				94, 77, 50, 84, 71, 41, 86, 77, 48, 78, 70, 44, 79, 73, 50, 77, 71, 50, 69, 59, 33, 89, 72, 42, 44, 35, 17, 63, 52, 32, 77, 62, 40, 93, 78, 52, 96, 82, 56, 93, 78, 56,
				80, 65, 46, 82, 67, 46, 81, 65, 37, 94, 74, 38, 120, 93, 62, 68, 37, 16, 79, 46, 26, 72, 38, 21, 72, 38, 19, 78, 46, 22, 150, 116, 69, 173, 126, 89, 84, 67, 39, 49, 50, 36,
				30, 19, 12, 48, 32, 29, 50, 31, 27, 100, 80, 67, 91, 75, 51, 77, 67, 40, 67, 60, 35, 63, 58, 36, 60, 56, 35, 73, 70, 47, 63, 57, 32, 78, 66, 40, 61, 54, 37, 77, 68, 48,
				85, 73, 52, 75, 64, 39, 84, 73, 49, 79, 67, 46, 73, 61, 43, 77, 66, 45, 83, 72, 43, 88, 71, 38, 98, 75, 49, 60, 33, 18, 86, 59, 42, 107, 80, 62, 102, 75, 58, 88, 62, 42,
				118, 88, 48, 156, 109, 74, 86, 66, 35, 61, 60, 41, 50, 41, 29, 64, 50, 43, 52, 34, 29, 68, 49, 35, 103, 87, 65, 85, 74, 52, 71, 64, 44, 67, 62, 44, 53, 51, 31, 90, 88, 61,
				82, 78, 53, 75, 69, 47, 48, 44, 27, 59, 52, 32, 85, 75, 54, 81, 73, 49, 74, 66, 43, 85, 76, 57, 80, 71, 54, 81, 74, 53, 88, 80, 51, 89, 76, 45, 88, 67, 46, 66, 42, 26,
				98, 78, 54, 102, 82, 58, 98, 79, 54, 96, 78, 52, 116, 92, 57, 143, 97, 64, 123, 100, 66, 82, 76, 54, 57, 49, 32, 49, 38, 25, 86, 70, 62, 58, 41, 25, 98, 83, 62, 86, 74, 56,
				59, 50, 34, 57, 50, 36, 65, 60, 41, 102, 98, 67, 79, 77, 51, 78, 79, 60, 58, 55, 37, 56, 51, 31, 78, 70, 49, 82, 75, 52, 79, 71, 49, 88, 79, 60, 85, 76, 59, 87, 81, 59,
				92, 86, 57, 92, 80, 51, 85, 65, 46, 67, 46, 29, 108, 93, 60, 95, 80, 47, 104, 88, 56, 83, 67, 36, 100, 82, 49, 133, 91, 59, 133, 107, 73, 94, 84, 57, 80, 70, 47, 60, 48, 30,
				96, 80, 67, 80, 63, 44, 85, 69, 48, 79, 66, 48, 61, 51, 34, 73, 66, 52, 86, 80, 61, 88, 83, 51, 59, 57, 31, 58, 59, 43, 66, 64, 43, 68, 65, 43, 69, 64, 42, 76, 68, 45,
				85, 75, 51, 89, 79, 56, 91, 81, 59, 91, 84, 60, 106, 101, 75, 98, 87, 60, 86, 69, 45, 71, 53, 27, 99, 87, 52, 98, 83, 53, 99, 81, 56, 77, 58, 35, 75, 59, 29, 118, 86, 56,
				113, 82, 51, 103, 83, 50, 102, 83, 50, 101, 85, 52, 81, 66, 37, 98, 78, 51, 98, 80, 56, 86, 76, 49, 100, 91, 60, 98, 90, 63, 88, 80, 58, 78, 72, 46, 77, 72, 47, 72, 68, 47,
				72, 67, 45, 82, 77, 55, 79, 74, 52, 82, 74, 49, 88, 78, 52, 86, 76, 51, 85, 75, 50, 84, 78, 52, 93, 88, 63, 84, 73, 46, 87, 69, 42, 79, 61, 31, 93, 82, 48, 88, 73, 44,
				78, 58, 37, 83, 62, 44, 96, 80, 51, 118, 92, 63, 108, 76, 47, 106, 82, 46, 112, 89, 52, 111, 94, 56, 94, 80, 46, 105, 84, 54, 121, 102, 77, 97, 88, 59, 101, 93, 57, 90, 80, 49,
				79, 70, 46, 86, 80, 56, 87, 81, 57, 87, 80, 56, 83, 73, 52, 87, 77, 56, 84, 74, 52, 92, 82, 57, 80, 70, 45, 83, 73, 48, 89, 80, 55, 98, 91, 65, 104, 98, 72, 102, 90, 63,
				93, 75, 47, 110, 91, 61, 102, 91, 58, 109, 93, 66, 114, 94, 72, 113, 92, 72, 108, 92, 65, 114, 89, 61, 117, 88, 58, 122, 98, 64, 122, 99, 64, 117, 100, 65, 114, 101, 68, 120, 99, 72,
				124, 106, 82, 104, 95, 67, 103, 94, 60, 100, 90, 61, 99, 90, 67, 99, 93, 70, 90, 84, 61, 81, 75, 52, 88, 72, 51, 90, 74, 52, 93, 77, 56, 94, 82, 58, 82, 72, 47, 81, 71, 46,
				95, 85, 60, 94, 87, 61, 96, 89, 61, 104, 91, 63, 108, 88, 59, 112, 92, 61, 110, 98, 68, 108, 92, 65, 124, 105, 82, 119, 98, 76, 105, 89, 62, 107, 85, 58, 117, 91, 62, 127, 104, 71,
				116, 94, 61, 124, 107, 75, 115, 101, 71, 111, 92, 66, 116, 98, 77, 107, 97, 70, 109, 100, 68, 106, 96, 68, 96, 87, 66, 80, 74, 53, 76, 70, 49, 82, 76, 55, 97, 78, 56, 94, 75, 53,
				93, 75, 53, 97, 85, 61, 96, 86, 61, 94, 84, 59, 96, 86, 61, 79, 71, 44, 78, 69, 40, 93, 78, 49, 105, 84, 53, 107, 86, 56, 98, 85, 58, 99, 83, 57, 106, 87, 62, 119, 98, 74,
				104, 88, 62, 104, 85, 58, 106, 85, 55, 122, 99, 67, 107, 85, 54, 112, 96, 66, 92, 81, 53, 80, 62, 39, 96, 78, 59, 77, 67, 42, 85, 76, 44, 84, 75, 48, 67, 57, 38, 54, 47, 28,
				63, 56, 37, 72, 65, 46};
		
	}
	
	public static char[] IMG2() {
		
		return new char[] {53, 65, 53, 54, 63, 52, 56, 60, 50, 54, 59, 50, 63, 70, 61, 78, 85, 76, 76, 83, 74, 63, 70, 61, 65, 71, 60, 73, 77, 65, 65, 66, 55, 74, 72, 63, 80, 78, 65, 87, 89, 69,
				94, 94, 77, 86, 86, 73, 78, 78, 67, 70, 71, 58, 71, 73, 59, 83, 84, 71, 90, 92, 79, 86, 88, 74, 80, 82, 68, 87, 89, 77, 68, 69, 61, 45, 46, 40, 48, 50, 47, 49, 53, 52,
				52, 56, 55, 47, 51, 50, 41, 45, 44, 24, 28, 27, 46, 59, 41, 53, 62, 45, 54, 59, 44, 57, 62, 49, 62, 70, 58, 67, 75, 63, 72, 80, 69, 72, 80, 68, 65, 73, 59, 74, 81, 66,
				77, 81, 67, 83, 83, 71, 83, 81, 65, 93, 91, 67, 108, 106, 85, 85, 82, 64, 80, 77, 61, 79, 80, 65, 77, 80, 65, 77, 80, 65, 83, 86, 71, 93, 96, 80, 87, 90, 73, 87, 89, 75,
				80, 82, 71, 48, 49, 41, 43, 46, 43, 46, 50, 49, 48, 52, 51, 42, 46, 45, 39, 43, 42, 28, 32, 31, 45, 59, 38, 50, 60, 41, 46, 52, 34, 54, 60, 44, 60, 68, 54, 76, 84, 70,
				90, 98, 84, 78, 87, 73, 66, 76, 60, 81, 91, 74, 90, 96, 81, 93, 95, 81, 86, 86, 68, 95, 92, 67, 93, 90, 67, 83, 79, 59, 85, 82, 64, 80, 83, 65, 80, 86, 67, 83, 89, 70,
				87, 93, 74, 86, 90, 72, 84, 87, 67, 82, 84, 68, 78, 80, 68, 55, 56, 47, 44, 47, 43, 44, 48, 47, 46, 50, 49, 38, 42, 41, 36, 40, 39, 29, 33, 32, 54, 67, 48, 49, 59, 41,
				48, 54, 38, 55, 61, 45, 60, 68, 52, 79, 88, 72, 88, 97, 81, 79, 88, 72, 75, 87, 68, 75, 87, 68, 87, 96, 78, 88, 93, 77, 99, 102, 83, 103, 105, 79, 77, 78, 55, 84, 85, 65,
				87, 88, 70, 81, 87, 67, 81, 90, 68, 86, 95, 72, 82, 91, 68, 71, 78, 56, 77, 80, 59, 82, 85, 66, 81, 84, 68, 69, 71, 59, 49, 52, 47, 42, 46, 45, 49, 53, 52, 42, 46, 45,
				35, 39, 38, 42, 46, 45, 59, 71, 57, 52, 61, 48, 63, 68, 57, 54, 59, 46, 51, 60, 44, 66, 75, 58, 80, 88, 71, 77, 87, 69, 71, 85, 64, 82, 96, 73, 86, 96, 75, 96, 103, 83,
				102, 108, 87, 100, 108, 85, 90, 98, 77, 90, 98, 81, 77, 84, 69, 76, 86, 65, 81, 93, 67, 81, 94, 68, 78, 90, 65, 91, 101, 76, 97, 101, 79, 95, 98, 77, 93, 96, 79, 76, 78, 64,
				46, 49, 43, 40, 44, 44, 45, 49, 48, 46, 50, 49, 36, 40, 39, 52, 56, 55, 59, 72, 62, 64, 73, 64, 72, 77, 69, 68, 73, 62, 63, 71, 52, 71, 79, 60, 74, 83, 64, 74, 84, 65,
				69, 83, 61, 90, 105, 83, 85, 97, 76, 85, 94, 74, 85, 95, 75, 95, 107, 87, 88, 100, 81, 77, 88, 72, 86, 98, 84, 76, 87, 66, 81, 91, 66, 94, 105, 80, 87, 99, 74, 85, 95, 70,
				90, 94, 71, 96, 100, 78, 86, 90, 71, 84, 87, 71, 44, 47, 40, 43, 47, 45, 42, 46, 44, 48, 52, 51, 43, 47, 46, 47, 51, 50, 54, 69, 60, 61, 73, 62, 64, 73, 61, 66, 72, 56,
				61, 66, 44, 64, 70, 50, 69, 77, 59, 72, 83, 65, 81, 95, 73, 91, 105, 86, 82, 94, 80, 82, 93, 77, 86, 98, 80, 80, 91, 74, 81, 88, 67, 69, 76, 55, 90, 101, 84, 94, 99, 81,
				95, 97, 75, 98, 103, 81, 90, 96, 74, 77, 85, 62, 74, 81, 58, 90, 96, 74, 88, 95, 73, 81, 88, 67, 53, 59, 47, 43, 47, 42, 43, 48, 43, 44, 48, 47, 41, 46, 46, 45, 52, 50,
				53, 68, 59, 54, 67, 55, 64, 74, 60, 77, 83, 65, 74, 77, 55, 76, 80, 60, 80, 88, 70, 56, 67, 50, 62, 76, 54, 92, 105, 87, 90, 99, 88, 86, 94, 79, 94, 103, 84, 86, 96, 76,
				87, 90, 65, 78, 81, 56, 87, 96, 76, 102, 105, 86, 96, 96, 75, 92, 95, 74, 93, 99, 77, 85, 94, 71, 82, 90, 67, 88, 96, 74, 98, 106, 83, 80, 88, 65, 67, 73, 59, 42, 47, 39,
				51, 56, 50, 50, 54, 52, 47, 52, 51, 45, 53, 50, 62, 77, 68, 59, 71, 60, 67, 77, 63, 80, 86, 68, 86, 89, 67, 90, 95, 74, 85, 93, 75, 61, 72, 55, 61, 73, 52, 79, 88, 71,
				83, 87, 76, 89, 89, 73, 105, 108, 86, 97, 104, 80, 88, 89, 60, 100, 101, 71, 92, 99, 75, 98, 102, 81, 100, 101, 80, 88, 94, 72, 97, 106, 82, 89, 99, 75, 87, 95, 72, 84, 92, 69,
				83, 91, 68, 72, 80, 57, 72, 78, 63, 45, 51, 41, 51, 57, 49, 54, 59, 54, 53, 58, 56, 48, 56, 53, 72, 87, 78, 56, 69, 57, 63, 73, 59, 79, 85, 67, 89, 92, 70, 89, 94, 74,
				75, 83, 66, 64, 75, 58, 72, 81, 62, 79, 83, 69, 80, 77, 66, 90, 82, 64, 128, 123, 99, 95, 97, 73, 108, 105, 74, 125, 122, 91, 111, 113, 88, 101, 104, 83, 96, 100, 78, 92, 100, 77,
				95, 106, 82, 86, 97, 73, 84, 93, 70, 75, 83, 60, 82, 90, 67, 77, 85, 62, 75, 82, 65, 49, 55, 43, 49, 55, 46, 54, 59, 53, 49, 54, 50, 46, 54, 51, 68, 83, 74, 57, 70, 58,
				58, 67, 53, 65, 71, 53, 82, 85, 63, 83, 87, 67, 75, 83, 65, 73, 84, 67, 68, 74, 57, 92, 92, 79, 95, 87, 75, 100, 84, 65, 122, 109, 86, 86, 84, 64, 106, 97, 71, 104, 96, 70,
				88, 86, 65, 90, 93, 73, 95, 101, 79, 90, 100, 76, 89, 102, 77, 77, 90, 65, 83, 92, 68, 87, 95, 72, 98, 106, 83, 84, 92, 70, 90, 97, 79, 65, 71, 57, 50, 56, 45, 50, 55, 48,
				54, 60, 54, 45, 54, 50, 68, 83, 74, 68, 81, 69, 61, 71, 57, 61, 67, 49, 76, 80, 57, 78, 83, 63, 73, 81, 63, 73, 84, 67, 83, 88, 72, 87, 84, 72, 100, 86, 75, 99, 76, 55,
				98, 79, 57, 102, 95, 82, 98, 85, 66, 83, 70, 52, 71, 65, 51, 85, 88, 69, 86, 95, 71, 80, 91, 67, 84, 98, 73, 83, 97, 71, 75, 83, 60, 89, 97, 74, 88, 96, 73, 89, 97, 74,
				93, 100, 81, 72, 78, 62, 48, 54, 41, 50, 55, 45, 60, 66, 59, 51, 59, 56, 65, 81, 70, 69, 83, 69, 64, 76, 60, 66, 74, 56, 71, 75, 56, 80, 85, 66, 81, 89, 70, 84, 94, 74,
				83, 89, 72, 74, 73, 60, 85, 76, 64, 100, 83, 65, 89, 74, 57, 72, 66, 56, 76, 64, 51, 84, 73, 58, 81, 78, 65, 101, 108, 90, 78, 89, 67, 77, 89, 66, 80, 93, 70, 81, 94, 70,
				79, 88, 63, 79, 87, 63, 83, 92, 67, 92, 100, 75, 81, 87, 66, 76, 81, 64, 55, 60, 47, 49, 54, 45, 53, 59, 53, 58, 65, 62, 61, 75, 62, 64, 78, 63, 64, 78, 60, 64, 75, 59,
				70, 76, 61, 71, 78, 60, 92, 100, 77, 104, 113, 88, 77, 86, 65, 71, 78, 61, 82, 84, 70, 96, 93, 80, 95, 92, 80, 54, 51, 43, 56, 47, 38, 79, 74, 60, 91, 96, 77, 98, 112, 90,
				84, 99, 77, 79, 92, 71, 80, 90, 70, 81, 90, 68, 90, 100, 72, 77, 87, 59, 87, 97, 69, 87, 96, 68, 78, 82, 59, 77, 78, 62, 58, 61, 49, 44, 48, 42, 45, 51, 47, 58, 63, 62,
				58, 67, 53, 61, 70, 56, 66, 75, 62, 66, 75, 62, 68, 74, 60, 66, 73, 55, 81, 88, 67, 86, 95, 70, 73, 84, 63, 71, 81, 64, 78, 85, 69, 85, 88, 74, 105, 105, 95, 62, 58, 54,
				61, 53, 45, 78, 74, 58, 94, 101, 76, 91, 105, 78, 79, 93, 68, 78, 89, 64, 83, 91, 67, 83, 90, 66, 84, 91, 64, 85, 93, 65, 91, 98, 71, 90, 97, 70, 74, 75, 53, 74, 73, 56,
				60, 61, 50, 48, 51, 46, 48, 53, 50, 50, 54, 53, 59, 63, 48, 57, 60, 48, 61, 64, 56, 67, 71, 60, 73, 80, 63, 76, 84, 65, 81, 89, 68, 87, 95, 73, 89, 100, 81, 68, 80, 63,
				88, 98, 79, 94, 99, 80, 116, 118, 104, 80, 74, 72, 69, 58, 50, 74, 69, 50, 82, 87, 57, 98, 110, 78, 75, 86, 56, 82, 89, 60, 88, 93, 65, 86, 89, 63, 75, 79, 55, 102, 105, 82,
				102, 105, 82, 82, 84, 61, 80, 79, 59, 87, 84, 69, 66, 65, 54, 49, 50, 45, 46, 49, 48, 50, 54, 53, 65, 68, 54, 63, 65, 53, 62, 64, 55, 65, 69, 56, 78, 86, 65, 78, 86, 65,
				77, 85, 65, 89, 96, 78, 89, 101, 83, 88, 101, 83, 90, 101, 79, 100, 109, 83, 113, 116, 98, 84, 76, 71, 88, 75, 66, 71, 64, 44, 66, 70, 40, 94, 102, 71, 77, 83, 53, 94, 98, 69,
				91, 92, 64, 89, 88, 63, 86, 84, 64, 91, 89, 69, 98, 96, 76, 82, 80, 60, 87, 82, 64, 74, 69, 53, 60, 57, 47, 44, 43, 39, 41, 42, 41, 49, 52, 52, 70, 76, 66, 73, 80, 66,
				65, 72, 55, 60, 68, 47, 69, 77, 53, 71, 79, 58, 69, 76, 58, 68, 75, 59, 76, 89, 73, 101, 116, 97, 94, 107, 82, 90, 101, 71, 112, 117, 90, 92, 84, 71, 79, 65, 52, 71, 62, 44,
				67, 67, 44, 79, 83, 58, 71, 73, 49, 80, 79, 56, 89, 84, 62, 93, 86, 67, 79, 72, 56, 71, 64, 48, 88, 81, 65, 89, 82, 67, 73, 65, 49, 54, 47, 32, 64, 58, 49, 51, 48, 45,
				43, 43, 42, 49, 52, 51, 73, 84, 79, 61, 74, 58, 63, 78, 51, 68, 81, 51, 69, 77, 51, 77, 84, 62, 74, 80, 64, 65, 72, 58, 58, 72, 56, 77, 94, 74, 87, 102, 76, 72, 87, 53,
				115, 123, 90, 129, 122, 97, 85, 71, 53, 69, 58, 42, 59, 57, 41, 62, 63, 46, 69, 66, 49, 67, 61, 45, 76, 67, 52, 77, 66, 53, 66, 56, 43, 68, 58, 45, 89, 79, 67, 94, 84, 72,
				68, 59, 44, 55, 47, 33, 67, 60, 52, 57, 53, 49, 45, 44, 44, 49, 52, 52, 70, 86, 82, 68, 84, 67, 76, 95, 59, 85, 100, 61, 68, 74, 49, 77, 79, 58, 74, 76, 58, 69, 75, 58,
				59, 69, 50, 70, 81, 60, 86, 97, 71, 68, 80, 47, 104, 112, 78, 130, 124, 95, 93, 80, 59, 70, 57, 44, 55, 50, 40, 61, 61, 42, 80, 78, 56, 59, 53, 36, 82, 73, 59, 73, 62, 50,
				61, 50, 39, 66, 57, 45, 76, 68, 54, 86, 79, 64, 64, 59, 45, 53, 48, 35, 66, 64, 52, 63, 61, 53, 41, 41, 37, 45, 47, 46, 74, 91, 83, 65, 79, 63, 79, 95, 56, 92, 104, 64,
				79, 82, 60, 81, 76, 58, 77, 74, 55, 76, 79, 58, 82, 85, 64, 87, 91, 69, 94, 100, 73, 88, 96, 65, 114, 120, 88, 120, 116, 87, 82, 70, 50, 74, 61, 49, 59, 52, 44, 71, 71, 50,
				99, 99, 71, 64, 60, 38, 81, 73, 55, 74, 65, 51, 55, 47, 37, 60, 56, 42, 69, 69, 51, 77, 79, 59, 62, 64, 50, 48, 50, 36, 64, 66, 49, 62, 64, 50, 44, 45, 38, 43, 43, 43,
				77, 87, 75, 73, 80, 65, 75, 81, 53, 84, 90, 58, 75, 77, 56, 84, 79, 60, 80, 76, 57, 77, 79, 58, 88, 91, 70, 102, 105, 83, 102, 106, 80, 111, 116, 87, 120, 123, 92, 110, 107, 77,
				80, 69, 48, 78, 65, 52, 63, 57, 45, 68, 69, 52, 85, 84, 66, 66, 62, 45, 102, 95, 76, 106, 99, 82, 48, 47, 33, 60, 62, 45, 72, 78, 56, 71, 80, 55, 73, 78, 62, 73, 75, 60,
				67, 70, 50, 54, 57, 40, 51, 53, 45, 43, 43, 43, 76, 78, 63, 75, 73, 58, 74, 70, 51, 73, 72, 51, 76, 78, 57, 93, 87, 69, 90, 86, 67, 76, 79, 58, 92, 96, 74, 100, 103, 81,
				98, 100, 74, 93, 93, 66, 109, 109, 80, 111, 108, 78, 98, 87, 65, 92, 81, 64, 72, 67, 52, 60, 61, 48, 71, 72, 59, 75, 72, 55, 86, 81, 60, 83, 80, 59, 51, 56, 39, 62, 71, 50,
				65, 78, 52, 65, 80, 52, 69, 76, 58, 68, 71, 54, 68, 71, 49, 66, 69, 49, 49, 50, 41, 43, 43, 42, 76, 74, 56, 80, 75, 57, 91, 83, 68, 88, 84, 68, 81, 83, 62, 83, 78, 60,
				88, 84, 65, 87, 90, 69, 94, 99, 77, 99, 103, 80, 107, 106, 81, 119, 114, 88, 127, 123, 95, 109, 105, 78, 101, 90, 66, 102, 91, 71, 100, 95, 77, 60, 63, 46, 79, 82, 67, 74, 75, 55,
				77, 75, 51, 74, 74, 51, 51, 58, 40, 60, 72, 49, 65, 80, 52, 63, 81, 50, 67, 75, 55, 70, 73, 54, 70, 74, 49, 70, 74, 52, 56, 58, 46, 38, 38, 37, 71, 69, 52, 87, 83, 61,
				86, 78, 61, 84, 80, 64, 74, 77, 56, 83, 78, 59, 85, 81, 62, 82, 85, 63, 80, 87, 64, 94, 97, 75, 114, 111, 88, 126, 116, 91, 126, 118, 91, 105, 101, 75, 108, 97, 73, 116, 105, 81,
				103, 100, 74, 55, 61, 37, 80, 88, 63, 88, 92, 65, 82, 82, 58, 67, 68, 46, 65, 71, 50, 70, 80, 56, 73, 87, 57, 67, 84, 51, 55, 63, 40, 82, 85, 64, 70, 75, 48, 64, 67, 44,
				62, 64, 50, 38, 38, 37, 71, 73, 57, 85, 85, 60, 81, 76, 55, 80, 78, 60, 78, 80, 59, 89, 84, 65, 83, 79, 59, 76, 79, 58, 70, 76, 54, 85, 88, 66, 118, 114, 91, 123, 111, 87,
				126, 115, 90, 112, 108, 83, 113, 103, 78, 127, 117, 89, 100, 98, 68, 84, 93, 58, 86, 96, 62, 99, 106, 71, 88, 91, 63, 59, 60, 36, 71, 74, 51, 78, 84, 59, 80, 91, 61, 81, 94, 60,
				63, 70, 45, 88, 92, 70, 78, 83, 56, 63, 67, 43, 59, 61, 47, 42, 42, 41, 65, 71, 57, 87, 88, 69, 88, 83, 65, 82, 76, 59, 88, 85, 66, 98, 94, 74, 76, 75, 53, 77, 80, 56,
				78, 78, 55, 83, 80, 57, 114, 110, 84, 121, 115, 86, 118, 109, 82, 128, 116, 92, 111, 102, 74, 116, 115, 80, 113, 120, 80, 105, 113, 74, 103, 109, 74, 96, 103, 64, 87, 93, 60, 78, 82, 54,
				73, 75, 48, 78, 82, 58, 75, 82, 54, 80, 90, 55, 64, 71, 42, 93, 97, 75, 71, 75, 54, 63, 66, 47, 59, 61, 47, 45, 46, 40, 58, 69, 53, 89, 92, 77, 93, 88, 73, 83, 76, 59,
				89, 84, 65, 93, 90, 70, 82, 83, 60, 82, 85, 61, 88, 85, 62, 97, 91, 67, 117, 113, 86, 120, 120, 88, 106, 100, 69, 123, 106, 80, 109, 101, 70, 86, 91, 54, 103, 117, 74, 97, 105, 71,
				90, 94, 61, 91, 98, 58, 84, 90, 57, 82, 86, 57, 79, 80, 50, 75, 78, 54, 69, 74, 48, 74, 84, 49, 57, 64, 32, 84, 88, 63, 75, 78, 57, 59, 62, 44, 64, 67, 52, 43, 45, 36,
				66, 79, 62, 75, 82, 66, 83, 84, 67, 89, 86, 68, 90, 87, 67, 89, 88, 68, 89, 91, 70, 82, 86, 65, 86, 85, 64, 104, 102, 78, 115, 113, 86, 111, 110, 81, 103, 97, 66, 121, 107, 75,
				104, 97, 64, 100, 104, 70, 102, 113, 78, 90, 102, 70, 83, 94, 58, 100, 107, 67, 107, 108, 76, 95, 93, 65, 80, 81, 50, 69, 72, 47, 63, 68, 43, 76, 85, 52, 75, 81, 50, 76, 80, 55,
				74, 77, 54, 53, 56, 37, 54, 57, 42, 47, 48, 39, 71, 83, 66, 74, 83, 66, 80, 85, 67, 79, 81, 61, 89, 87, 67, 100, 101, 81, 91, 95, 77, 83, 89, 72, 83, 87, 67, 94, 95, 73,
				109, 109, 84, 106, 104, 77, 104, 98, 67, 128, 118, 82, 109, 103, 70, 102, 103, 73, 103, 110, 82, 80, 94, 64, 86, 101, 64, 99, 105, 66, 115, 113, 81, 96, 90, 62, 83, 84, 49, 72, 75, 48,
				60, 65, 41, 71, 80, 50, 82, 89, 57, 72, 77, 48, 71, 75, 50, 51, 54, 33, 46, 49, 34, 49, 50, 41, 75, 82, 67, 79, 85, 69, 81, 85, 67, 79, 81, 60, 90, 91, 69, 97, 100, 81,
				90, 96, 81, 91, 99, 86, 90, 99, 80, 86, 92, 70, 94, 96, 73, 94, 92, 67, 108, 102, 74, 115, 108, 75, 107, 102, 71, 94, 94, 66, 99, 103, 77, 79, 92, 62, 84, 98, 61, 99, 106, 66,
				111, 109, 77, 92, 88, 58, 103, 105, 67, 94, 98, 70, 65, 70, 47, 69, 77, 50, 78, 85, 53, 66, 71, 39, 71, 75, 48, 61, 65, 41, 64, 67, 48, 48, 50, 41, 85, 83, 71, 85, 84, 71,
				86, 86, 70, 85, 86, 66, 88, 92, 68, 91, 96, 77, 87, 94, 81, 89, 98, 89, 88, 99, 83, 88, 98, 77, 99, 103, 81, 102, 99, 77, 106, 99, 75, 99, 92, 67, 101, 97, 70, 103, 102, 75,
				104, 105, 77, 85, 91, 62, 86, 95, 60, 100, 106, 67, 104, 107, 74, 93, 94, 62, 112, 114, 74, 84, 88, 59, 75, 79, 58, 82, 89, 65, 79, 86, 55, 70, 75, 42, 71, 76, 46, 61, 65, 39,
				64, 67, 46, 49, 50, 41};
		
		
	}

}
