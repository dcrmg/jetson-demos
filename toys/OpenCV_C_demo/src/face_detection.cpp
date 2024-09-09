#include <opencv2/opencv.hpp>

int main(int argc,char *argv[])
{
    // load face classifier
    cv::CascadeClassifier faceCascade;
    faceCascade.load(argv[3]);

    cv::Mat image = cv::imread(argv[1]);

    std::vector<cv::Rect> faces;
	faceCascade.detectMultiScale(image, faces, 1.1, 3, 0, cv::Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		cv::rectangle(image, faces[i], cv::Scalar(0, 255, 0), 2);
	}

	cv::imwrite(argv[2], image);

    return 0;
}