#include "lab1.h"

#include <cstdio>
#include <cassert>
#include <sndfile.h>

#include "SyncedMemory.h"

namespace {

const unsigned W = 1280, H = 720, FPS = 24;
const char *AUDIO_FILE = "audio.wav";
const unsigned SAMPLE_RATE = 44100, CHANNELS = 2;
const unsigned WINDOW_SIZE = SAMPLE_RATE / FPS;
const unsigned BLOCKDIM_X = 32, BLOCKDIM_Y = 24;

}  // namespace

namespace kernel {

__global__ void GetAudioWindow(const float *audio, unsigned offset,
		float *window) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(x < WINDOW_SIZE))
		return;

	window[x] = audio[offset + 2*x];
	window[WINDOW_SIZE + x] = audio[offset + 2*x + 1];
}

__device__ int myabs(int x) {
	return x < 0 ? -x : x;
}

__device__ uint8_t clip(int x) {
	return x > 255 ? 255 : x < 0 ? 0 : x;
}

__global__ void GenerateFrame(const float *window, unsigned curFrame,
		uint8_t *yuv) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (!(x < W && y < H))
		return;

	size_t off = x * WINDOW_SIZE / W + (y >= H/2) * WINDOW_SIZE;
	if (abs(window[off] - (2.0 * (y % (H/2)) / (H/2 - 1) - 1)) < 0.02)
		yuv[y*W + x] = 255;
	else
		yuv[y*W + x] = 153;

	if (x % 2 == 0 && y % 2 == 0) {
		yuv[W*H + (y/2)*(W/2) + x/2] = 221;
		yuv[W*H + (W/2)*(H/2) + (y/2)*(W/2) + x/2] =
			clip(256 - myabs((curFrame + x + y/2) % 512 - 256));
	}
}

}  // namespace kernel

class Lab1VideoGenerator::Impl {
public:
	Impl();
	void get_info(Lab1VideoInfo&);
	void Generate(uint8_t *yuv);
private:
	MemoryBuffer<float> m_buffer;
	SyncedMemory<float> m_audioMem, m_windowMem;
	unsigned m_numFrames;
	unsigned m_curFrame;
};

Lab1VideoGenerator::Impl::Impl() {
	SF_INFO sfinfo;
	SNDFILE *sndfile = sf_open(AUDIO_FILE, SFM_READ, &sfinfo);
	assert(sfinfo.samplerate == SAMPLE_RATE);
	assert(sfinfo.channels == CHANNELS);
	m_numFrames = sfinfo.frames * FPS / SAMPLE_RATE;

	printf("numFrames = %u, length = %u:%u.%u\n",
			m_numFrames, m_numFrames / FPS / 60,
			m_numFrames / FPS % 60, m_numFrames % FPS);

	m_buffer.Realloc(sfinfo.frames * CHANNELS + WINDOW_SIZE * CHANNELS);
	m_audioMem = m_buffer.CreateSync(sfinfo.frames * CHANNELS);
	m_windowMem = m_buffer.CreateSync(WINDOW_SIZE * CHANNELS,
			sfinfo.frames * CHANNELS);

	assert(sf_readf_float(sndfile, m_audioMem.get_cpu_wo(), sfinfo.frames)
			== sfinfo.frames);
	sf_close(sndfile);
}

void Lab1VideoGenerator::Impl::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = m_numFrames;
	info.fps_n = FPS;
	info.fps_d = 1;
}

void Lab1VideoGenerator::Impl::Generate(uint8_t *yuv) {
	assert(m_curFrame < m_numFrames);
	kernel::GetAudioWindow<<<
		(WINDOW_SIZE - 1)/BLOCKDIM_X + 1,
		BLOCKDIM_X>>>(
			m_audioMem.get_gpu_ro(),
			m_curFrame * SAMPLE_RATE * CHANNELS / FPS,
			m_windowMem.get_gpu_wo());
	kernel::GenerateFrame<<<
		dim3((W - 1)/BLOCKDIM_X + 1, (H - 1)/BLOCKDIM_Y + 1),
		dim3(BLOCKDIM_X, BLOCKDIM_Y)>>>(m_windowMem.get_gpu_ro(),
				m_curFrame, yuv);
	m_curFrame++;
}

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	impl->get_info(info);
};

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	impl->Generate(yuv);
}
