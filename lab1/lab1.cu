#include "lab1.h"

namespace {

const unsigned W = 640;
const unsigned H = 480;
const unsigned NFRAME = 240;

}  // namespace

class Lab1VideoGenerator::Impl {
public:
	Impl(): t(0) {}
	void get_info(Lab1VideoInfo&);
	void Generate(uint8_t *yuv);
private:
	int t;
};

void Lab1VideoGenerator::Impl::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
}

void Lab1VideoGenerator::Impl::Generate(uint8_t *yuv) {
	cudaMemset(yuv, t*255/NFRAME, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
	++t;
}

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	impl->get_info(info);
};

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	impl->Generate(yuv);
}
