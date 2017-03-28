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
const unsigned WAVE_POINT_BUFSIZE = W * CHANNELS / 2;
const float SPEED = 5.0;
const float PI = acos(-1);
const unsigned BLOCKDIM_X = 32, BLOCKDIM_Y = 24;

}  // namespace

namespace kernel {

__global__ void GetAudioWindowSum(const float *audio, float *windowSum) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned tid = threadIdx.x;
	__shared__ float buf[CHANNELS][BLOCKDIM_X];

	for (unsigned c = 0; c < CHANNELS; c++) {
		if (x < WINDOW_SIZE)
			buf[c][tid] = audio[x * CHANNELS + c];
		else
			buf[c][tid] = 0.0;
		buf[c][tid] *= buf[c][tid];
	}

	for (unsigned i = BLOCKDIM_X >> 1; i > 0; i >>= 1) {
		__syncthreads();
		if (tid < i) {
			for (unsigned c = 0; c < CHANNELS; c++)
				buf[c][tid] += buf[c][tid + i];
		}
	}

	__syncthreads();
	if (tid < CHANNELS)
		atomicAdd(&windowSum[tid], buf[tid][0]);
}

__device__ int myabs(int x) {
	return x < 0 ? -x : x;
}

__device__ uint8_t clip(int x) {
	return x > 255 ? 255 : x < 0 ? 0 : x;
}

__global__ void GenerateFrame(unsigned curFrame, const float *wavePoint,
		uint8_t *yuv) {
	const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (!(x < W && y < H))
		return;

	float p = 1e-9;
	for (unsigned i = 0; i < WAVE_POINT_BUFSIZE; i++) {
		float px, py, pp, r;
		px = wavePoint[4 * i];
		py = wavePoint[4 * i + 1];
		pp = wavePoint[4 * i + 2];
		r = SPEED * (curFrame - wavePoint[4 * i + 3]);

		float dist = abs(sqrt((x-px)*(x-px) + (y-py)*(y-py)) - sqrt(r*r));
		if (dist <= SPEED)
			p += pp * (1 - dist / SPEED);
	}
	yuv[y*W + x] = clip(32 + 192 * p);

	if (x % 2 == 0 && y % 2 == 0) {
		yuv[W*H + (y/2)*(W/2) + x/2] = 221;
		yuv[W*H + (W/2)*(H/2) + (y/2)*(W/2) + x/2] =
			clip(48 + 160 - myabs((curFrame + x + y/2) % 320 - 160));
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
	SyncedMemory<float> m_audioMem, m_windowSumMem, m_wavePointMem;
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

	m_buffer.Realloc(sfinfo.frames * CHANNELS + CHANNELS +
			4 * WAVE_POINT_BUFSIZE);
	m_audioMem = m_buffer.CreateSync(sfinfo.frames * CHANNELS);
	m_windowSumMem = m_buffer.CreateSync(CHANNELS,
			sfinfo.frames * CHANNELS);
	m_wavePointMem = m_buffer.CreateSync(4 * WAVE_POINT_BUFSIZE,
			sfinfo.frames * CHANNELS + CHANNELS);

	assert(sf_readf_float(sndfile, m_audioMem.get_cpu_wo(), sfinfo.frames)
			== sfinfo.frames);
	sf_close(sndfile);

	float *wavePoint = m_wavePointMem.get_cpu_wo();
	memset(wavePoint, 0, 4 * WAVE_POINT_BUFSIZE * sizeof(float));
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

	float *windowSum = m_windowSumMem.get_cpu_wo();
	windowSum[0] = windowSum[1] = 0.0;

	unsigned offset = m_curFrame * SAMPLE_RATE * CHANNELS / FPS;
	kernel::GetAudioWindowSum<<<
		(WINDOW_SIZE - 1)/BLOCKDIM_X + 1, BLOCKDIM_X,
		CHANNELS * BLOCKDIM_X * sizeof(float)>>>(
			m_audioMem.get_gpu_ro() + offset,
			m_windowSumMem.get_gpu_rw());

	m_windowSumMem.get_cpu_ro();
	float *wavePoint = m_wavePointMem.get_cpu_wo();
	offset = 4 * (m_curFrame * CHANNELS % WAVE_POINT_BUFSIZE);
	for (unsigned c = 0; c < CHANNELS; c++) {
		wavePoint[offset + 4*c] =
			W / 2 - cos(m_curFrame / 100.0 + c * PI) * W / 4;
		wavePoint[offset + 4*c + 1] =
			H / 2 + sin(m_curFrame / 100.0 + c * PI) * H / 4;
		wavePoint[offset + 4*c + 2] = sqrt(windowSum[c] / WINDOW_SIZE);
		wavePoint[offset + 4*c + 3] = m_curFrame;
	}

	kernel::GenerateFrame<<<
		dim3((W - 1)/BLOCKDIM_X + 1, (H - 1)/BLOCKDIM_Y + 1),
		dim3(BLOCKDIM_X, BLOCKDIM_Y)>>>(m_curFrame,
				m_wavePointMem.get_gpu_ro(), yuv);
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
