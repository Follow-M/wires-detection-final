#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

String filename; //name of a source image
Mat src, mask; //src - source image; mask - includes one line that count_av_color() function works at a time
vector<Vec4i> lines; //detected by HoughLinesP lines on a binarised image


int count_av_color(int x, int y)
{
	//dilating the line we are forking with so not to count unnecessary adjacent pixels	
	Mat elem = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
	Mat dilated;
	dilate(mask, dilated, elem);

	//storing two massives of colors
	Point3i m1(0, 0, 0), m2(0, 0, 0);
	int m1Count = 0, m2Count = 0;

	int kernel = 3;

	//boolean variables to detect when to change arrays
	bool line = false, change_diff = false, change_same = false, first_array = true;

	int i, j;

	//threshold to exclude lines
	int threshold = 90, color_variance;

	for (j = y - kernel; j <= y + kernel; j++)
	{
		//checking if pixels are in the range of image
		if (j < 0 || j > src.rows)
			continue;

		if (j % 2 == 0)
		{
			for (i = x - kernel; i <= x + kernel; i++)
			{
				if (i<0 || i > src.cols)
					continue;
				if ((int)dilated.at<uchar>(j, i) == 0 && line &&
					(change_diff || !(change_diff || change_same)))
				{
					change_diff = false;
					change_same = false;
					//swaping arrays
					if (first_array)
						first_array = false;
					else
						first_array = true;
				}
				if ((int)dilated.at<uchar>(j, i) == 0)
					line = false;
				else line = true;

				if (!line)
				{
					//adding pixel values to the necessary array
					change_same = false;
					if (first_array)
					{
						m1.x += (int)src.at<Vec3b>(Point(i, j))[0];
						m1.y += (int)src.at<Vec3b>(Point(i, j))[1];
						m1.z += (int)src.at<Vec3b>(Point(i, j))[2];
						m1Count++;
					}
					else
					{
						m2.x += (int)src.at<Vec3b>(Point(i, j))[0];
						m2.y += (int)src.at<Vec3b>(Point(i, j))[1];
						m2.z += (int)src.at<Vec3b>(Point(i, j))[2];
						m2Count++;
					}
				}
			}
			i--;
		}
		else
		{
			//repeating all the previous operations for odd lines
			for (i = x + kernel; i >= x - kernel; i--)
			{
				if (i<0 || i > src.cols)
					continue;

				if ((int)dilated.at<uchar>(j, i) == 0 && line &&
					(change_diff || !(change_diff || change_same)))
				{
					change_diff = false;
					change_same = false;
					if (first_array)
						first_array = false;
					else
						first_array = true;
				}
				if ((int)dilated.at<uchar>(j, i) == 0)
					line = false;
				else line = true;

				if (!line)
				{
					change_same = false;
					if (first_array)
					{
						m1.x += (int)src.at<Vec3b>(Point(i, j))[0];
						m1.y += (int)src.at<Vec3b>(Point(i, j))[1];
						m1.z += (int)src.at<Vec3b>(Point(i, j))[2];
						m1Count++;
					}
					else
					{
						m2.x += (int)src.at<Vec3b>(Point(i, j))[0];
						m2.y += (int)src.at<Vec3b>(Point(i, j))[1];
						m2.z += (int)src.at<Vec3b>(Point(i, j))[2];
						m2Count++;
					}
				}
			}
			i++;
		}
		change_same = false;
		change_diff = false;

		if (((int)dilated.at<uchar>(j + 1, i) != 0 && line) || ((int)dilated.at<uchar>(j + 1, i) == 0 && !line))
			change_same = true;
		else
			change_diff = true;
	}
	//getting average pixel colors
	m1.x /= m1Count;
	m1.y /= m1Count;
	m1.z /= m1Count;

	m2.x /= m2Count;
	m2.y /= m2Count;
	m2.z /= m2Count;

	//computing color variance and comparing with threshold
	color_variance = (int)(pow(m1.x - m2.x, 2) + pow(m1.y - m2.y, 2) + pow(m1.z - m2.z, 2));

	if (color_variance <= threshold)
		return 1;
	else
		return 0;
}


int backgr_check(int num)
{
	//beginning, end and center of the line
	Point a, b, c;
	a.x = lines[num][0];
	a.y = lines[num][1];
	b.x = lines[num][2];
	b.y = lines[num][3];

	//define center of the line segment
	c.x = (a.x + b.x) / 2;
	c.y = (a.y + b.y) / 2;

	//define to additional points for checking (lambda indicates the relation between parts)
	Point d, e;
	int lambda = 5;

	d.x = (b.x + lambda*a.x) / (1 + lambda);
	d.y = (b.y + lambda*a.y) / (1 + lambda);
	e.x = (a.x + lambda*b.x) / (1 + lambda);
	e.y = (a.y + lambda*b.y) / (1 + lambda);

	//checking if the line is on the same-color background
	int res = 0;
	res = count_av_color(d.x, d.y) + count_av_color(c.x, c.y) + count_av_color(e.x, e.y);

	if (res > 0)
		return 0;
	else
		return -1;
}


int main(const int argc, const char * argv[]) {
	//opening initial source image
	if (argc == 2)
		filename = argv[1];
	else
		filename = "1.jpg";
	src = imread(filename, IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "No image supplied" << endl;
		return -1;
	}

	//changing resolution of image if necessary
	bool dilate_im = true;
	if (src.cols > 1500 || src.rows > 1500)
	{
		resize(src, src, Size(), 0.25, 0.25, INTER_LINEAR);
		bool dilate_im = false;
	}
	namedWindow("Initial photo", WINDOW_AUTOSIZE);
	imshow("Initial photo", src);
	waitKey(0);

	//blurring image to smooth all the noise if it is of poor quality
	Mat src_blurred;
	if (dilate_im)
	{
		GaussianBlur(src, src_blurred, Size(3, 3), 0, 0, BORDER_DEFAULT);
	}
	else
		src_blurred = src.clone();

	//morphological transformations (Black Hat: source image - closed image)
	Mat result_gray, binary;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));

	Mat src_blurred_blackHat, result_gray_blurred;

	morphologyEx(src_blurred, src_blurred_blackHat, MORPH_BLACKHAT, element);

	cvtColor(src_blurred_blackHat, result_gray, COLOR_BGR2GRAY);

	GaussianBlur(result_gray, result_gray_blurred, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//binarising image by OTSU
	threshold(result_gray_blurred, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	Mat inpaintMask = Mat::zeros(src.size(), CV_8U);
	Mat res;

	//detecting lines
	HoughLinesP(binary, lines, 1, CV_PI / 180, 80, 30, 4);
	for (size_t i = 0; i < lines.size(); i++)
	{
		mask = Mat::zeros(src.size(), CV_8U);
		line(mask, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255, 255, 255), 1, CV_AA);
		if (!backgr_check(i))
		{
			//adding line to the mask if it is on the same-color background
			line(src_blurred, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, CV_AA);
			line(inpaintMask, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255, 255, 255), 1, CV_AA);
		}

	}
	namedWindow("Detected wires", WINDOW_NORMAL);
	imshow("Detected wires", src_blurred);
	waitKey(0);

	//dilating inPaintMask if image was of low resolution
	if (dilate_im)
	{
		Mat elem = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
		dilate(inpaintMask, inpaintMask, elem);
	}

	//deleting wires
	inpaint(src, inpaintMask, src, 5, INPAINT_TELEA);

	//save the result and show it
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);

	imwrite("filtered image.jpg", src, compression_params);
	namedWindow("Resulted image", WINDOW_AUTOSIZE);
	imshow("Resulted image", src);
	waitKey(0);



	return 0;
}
