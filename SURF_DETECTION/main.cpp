#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include<iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) {


	Mat src1 = imread("E:/photo/1.jpg", 1);
	Mat src2 = imread("E:/photo/2.jpg", 1);
	if (!src1.data || !src2.data) {
		printf("could not load image...\n");
		return -1;
	}
	//imshow("photo 1", src1);
	//imshow("photo 2", src2);

	int minHessian = 500;
	Ptr<SURF> detector = SURF::create(minHessian);//����SURF��Hessian��ֵ������������
	std::vector<KeyPoint> keypoints_1, keypoints_2;//vectorģ�������ܹ�����������͵Ķ�̬���飬�ܹ����Ӻ�ѹ������

	/*����detect��������SURF�����ؼ��㣬�����vector����*/
	Mat dst1, dst2;
	detector->detectAndCompute(src1, Mat(), keypoints_1, dst1);
	detector->detectAndCompute(src2, Mat(), keypoints_2, dst2);

	/*���������ؼ���*/
	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints(src1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(src2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	Ptr<DescriptorMatcher>matcher = DescriptorMatcher::create("FlannBased");
	vector<DMatch>match;

	matcher->match(dst1, dst2, match);
	double Max_dist = 0;
	double Min_dist = 100;
	for (int i = 0; i < dst1.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < Min_dist)Min_dist = dist;
		if (dist > Max_dist)Max_dist = dist;
	}
	cout << "��̾���" << Min_dist << endl;
	cout << "�����" << Max_dist << endl;

	vector<DMatch>goodmaches;
	for (int i = 0; i < dst1.rows; i++)
	{
		if (match[i].distance < 2 * Min_dist)
			goodmaches.push_back(match[i]);
	}
	Mat img_maches;
	drawMatches(src1, keypoints_1, src2, keypoints_2, goodmaches, img_maches);

	for (int i = 0; i < goodmaches.size(); i++)
	{
		cout << "����������ƥ�䣺" << goodmaches[i].queryIdx << "--" << goodmaches[i].trainIdx << endl;
	}
	//const char* win1 = "Ч��ͼ1";
	//const char* win2 = "Ч��ͼ2";
	const char* win3 = "ƥ��Ч��";
	//namedWindow(win1, WINDOW_NORMAL);
	//namedWindow(win2, WINDOW_NORMAL);
	namedWindow(win3, WINDOW_FULLSCREEN);
	//imshow(win1, img_keypoints_1);
	//imshow(win2, img_keypoints_2);
	imshow(win3, img_maches);
	waitKey(0);
	return 0;
}