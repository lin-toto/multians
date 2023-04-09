/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "multians.h"

// encoder configuration //
#define NUM_SYMBOLS 256
#define NUM_STATES 1024

// seed for PRNG to generate random test data
#define SEED 5

// decoder configuration //

// SUBSEQUENCE_SIZE must be a multiple of 4
#define SUBSEQUENCE_SIZE 4

// number of GPU threads per thread block //
#define THREADS_PER_BLOCK 128


std::string readFile(const std::string &name, std::ios::ios_base::openmode mode) {
    std::ifstream file(name, mode);
    if (!file.good()) [[unlikely]] throw std::runtime_error("Error reading strings file");

    std::stringstream strStream;
    strStream << file.rdbuf();
    std::string str = strStream.str();

    return str;
}

template<class ValueType>
std::vector<ValueType> stringToSymbols(const std::string &str) {
    std::vector<ValueType> symbols;
    symbols.resize(str.length());
    std::transform(str.begin(), str.end(), symbols.begin(), [] (char v) {
        return static_cast<unsigned char>(v);
    });

    return symbols;
}

std::string jsonOutput(bool correct, unsigned int nSplit, size_t originalSize, size_t compressedSize, unsigned int elapsed) {
    const char *jsonLiteral = R"({
    "result_correct": %s,
    "n_splits": %u,
    "original_size_bytes": %llu,
    "compressed_size_bytes": %llu,
    "time": %u,
    "throughput_mb": %.2f
})";
    float throughput = originalSize / (elapsed / 1000000.0) / 1024 / 1024;

    char buf[200]; // Probably more than enough?
    snprintf(buf, 200, jsonLiteral, correct ? "true" : "false", nSplit, originalSize, compressedSize, elapsed, throughput);

    return buf;
}

void run_file(long int num_threads, std::string fileName) {
    // vectors to record timings
    std::vector<std::pair<std::string, size_t>> timings_cuda;
    std::vector<std::pair<std::string, size_t>> timings_multicore;


    auto textBuf = stringToSymbols<uint8_t>(readFile(fileName, std::ios_base::in));
    auto dist = ANSTableGenerator::generate_distribution_from_buffer(
            SEED, NUM_STATES, textBuf.data(), textBuf.size());

    // create an ANS table, based on the distribution
    auto table = ANSTableGenerator::generate_table(
            dist.prob, dist.dist, dist.symbols, NUM_SYMBOLS,
            NUM_STATES);

    // derive an encoder table from the ANS table
    auto encoder_table = ANSTableGenerator::generate_encoder_table(table);

    // derive a decoder table from the ANS table
    auto decoder_table
            = ANSTableGenerator::get_decoder_table(encoder_table);

    // tANS-encode the generated data using the encoder table
    auto input_buffer = ANSEncoder::encode(
            textBuf.data(), textBuf.size(), encoder_table);

    // allocate a buffer for the decoded output
    auto output_buffer = std::make_shared<CUHDOutputBuffer>(textBuf.size());

#ifdef CUDA

    // in GPU DRAM, allocate buffers for compressed input, coding table
    // and decompressed output
    auto gpu_in_buf
        = std::make_shared<cuhd::CUHDGPUInputBuffer>(input_buffer);
    auto gpu_table
        = std::make_shared<cuhd::CUHDGPUCodetable>(decoder_table);
    auto gpu_out_buf
        = std::make_shared<cuhd::CUHDGPUOutputBuffer>(output_buffer);

    // allocate auxiliary memory
    auto gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
        input_buffer->get_compressed_size(),
        SUBSEQUENCE_SIZE, THREADS_PER_BLOCK);

    gpu_table->allocate();
    gpu_in_buf->allocate();
    gpu_out_buf->allocate();
    gpu_decoder_memory->allocate();

    // copy decoder table to the GPU
    gpu_table->cpy_host_to_device();

    // copy compressed input to the GPU
    gpu_in_buf->cpy_host_to_device();

    // decode the compressed data on the GPU
    TIMER_START(timings_cuda, "gpu")
    cuhd::CUHDGPUDecoder::decode(
        gpu_in_buf, input_buffer->get_compressed_size(),
        gpu_out_buf, output_buffer->get_uncompressed_size(),
        gpu_table, gpu_decoder_memory, input_buffer->get_first_state(),
        input_buffer->get_first_bit(), decoder_table->get_num_entries(),
        16, SUBSEQUENCE_SIZE, THREADS_PER_BLOCK);
    TIMER_STOP

    // copy decompressed output from the GPU to the host system
    gpu_out_buf->cpy_device_to_host();

    // reverse all bytes
    output_buffer->reverse();

    // check for errors in decompressed data
    bool correct = cuhd::CUHDUtil::equals(textBuf.data(),
        output_buffer->get_decompressed_data().get(), textBuf.size());
#endif

    std::cout << jsonOutput(correct, 0, output_buffer->get_uncompressed_size(), input_buffer->get_compressed_size() * sizeof(UNIT_TYPE), 
            timings_cuda.at(0).second);

}

void run(long int input_size, long int num_threads) {

    // print column headers
    std::cout << "\u03BB | compressed size (bytes) | ";
    #ifdef MULTI
    std::cout << "time [multicore] (\u03BCs) | ";
    #endif
    #ifdef CUDA
    std::cout << "time [gpu decode] (\u03BCs)";
    #endif
    std::cout << std::endl << std::endl;
    
    for(float lambda = 0.1f; lambda < 2.5f; lambda += 0.16) {
    
        // vectors to record timings
        std::vector<std::pair<std::string, size_t>> timings_cuda;
        std::vector<std::pair<std::string, size_t>> timings_multicore;
        
        std::cout << std::left << std::setw(5) << lambda << std::setfill(' ');
        
        // generate random, exponentially distributed data
        auto dist = ANSTableGenerator::generate_distribution(
            SEED, NUM_SYMBOLS, NUM_STATES,
            [&](double x) {return lambda * exp(-lambda * x);});

        auto random_data = 
            ANSTableGenerator::generate_test_data(
                dist.dist, input_size, NUM_STATES, SEED);
        
        // create an ANS table, based on the distribution
        auto table = ANSTableGenerator::generate_table(
            dist.prob, dist.dist, nullptr, NUM_SYMBOLS,
            NUM_STATES);

        // derive an encoder table from the ANS table
        auto encoder_table = ANSTableGenerator::generate_encoder_table(table);
        
        // derive a decoder table from the ANS table
        auto decoder_table
            = ANSTableGenerator::get_decoder_table(encoder_table);
        
        // tANS-encode the generated data using the encoder table
        auto input_buffer = ANSEncoder::encode(
            random_data->data(), input_size, encoder_table);
        
        // allocate a buffer for the decoded output
        auto output_buffer = std::make_shared<CUHDOutputBuffer>(input_size);
        
        #ifdef CUDA
        
        // in GPU DRAM, allocate buffers for compressed input, coding table
        // and decompressed output
        auto gpu_in_buf
            = std::make_shared<cuhd::CUHDGPUInputBuffer>(input_buffer);
        auto gpu_table
            = std::make_shared<cuhd::CUHDGPUCodetable>(decoder_table);
        auto gpu_out_buf
            = std::make_shared<cuhd::CUHDGPUOutputBuffer>(output_buffer);
        
        // allocate auxiliary memory
        auto gpu_decoder_memory = std::make_shared<cuhd::CUHDGPUDecoderMemory>(
            input_buffer->get_compressed_size(),
            SUBSEQUENCE_SIZE, THREADS_PER_BLOCK);
        
        gpu_table->allocate();
        gpu_in_buf->allocate();
        gpu_out_buf->allocate();
        gpu_decoder_memory->allocate();
        
        // copy decoder table to the GPU
        gpu_table->cpy_host_to_device();
        
        // copy compressed input to the GPU
        gpu_in_buf->cpy_host_to_device();
        
        // decode the compressed data on the GPU
        TIMER_START(timings_cuda, "gpu")
        cuhd::CUHDGPUDecoder::decode(
            gpu_in_buf, input_buffer->get_compressed_size(),
            gpu_out_buf, output_buffer->get_uncompressed_size(),
            gpu_table, gpu_decoder_memory, input_buffer->get_first_state(),
            input_buffer->get_first_bit(), decoder_table->get_num_entries(),
            11, SUBSEQUENCE_SIZE, THREADS_PER_BLOCK);
        TIMER_STOP
        
        // copy decompressed output from the GPU to the host system
        gpu_out_buf->cpy_device_to_host();
        
        // reverse all bytes
        output_buffer->reverse();
        
        // check for errors in decompressed data
        if(cuhd::CUHDUtil::equals(random_data->data(),
            output_buffer->get_decompressed_data().get(), input_size));
        else std::cout << "mismatch" << std::endl;
        #endif
        
        #ifdef MULTI
        
        // decode the compressed data with multiple CPU-threads
        TIMER_START(timings_multicore, "multicore")
        MulticoreDecoder::decode(SUBSEQUENCE_SIZE, num_threads,
            input_buffer->get_compressed_size(),
            output_buffer, input_buffer, decoder_table);
        TIMER_STOP
        
        // reverse all bytes
        output_buffer->reverse();
        
        // check for errors in decompressed data
        if(cuhd::CUHDUtil::equals(random_data->data(),
            output_buffer->get_decompressed_data().get(), input_size));
        else std::cout << "mismatch" << std::endl;
        #endif
        
        // print compressed size (bytes)
        std::cout << std::left << std::setw(10)
            << input_buffer->get_compressed_size() * sizeof(UNIT_TYPE)
            << std::setfill(' ');
        
        // print multicore runtime
        #ifdef MULTI
        std::cout << std::left << std::setw(10)
            << timings_multicore.at(0).second;
        #endif
        
        // print GPU runtime
        #ifdef CUDA
        std::cout << std::left << std::setw(10)
            <<  timings_cuda.at(0).second << std::endl;
        #endif
    }
}

int main(int argc, char **argv) {

    // name of the binary file
    const char* bin = argv[0];
    
    auto print_help = [&]() {
        std::cout << "USAGE: " << bin << " <compute device index> "
            << "<size of input in megabytes> "
            << "<number of CPU threads> "
            << "[file name]"<< std::endl;
    };

    if(argc < 4) {print_help(); return 1;}
    
    // compute device to use
    const std::int64_t compute_device_id = atoi(argv[1]);
    
    // input size in MB
    const long int size = atoi(argv[2]) * 1024 * 1024;
    
    // number of CPU threads
    const long int threads = atoi(argv[3]);
    
    if(compute_device_id < 0 || size < 1 || threads < 1) {
        print_help();
        return 1;
    }
    
    // SUBSEQUENCE_SIZE must be a multiple of 4
    assert(SUBSEQUENCE_SIZE % 4 == 0);
    
    // select the GPU to be used for decompression
	#ifdef CUDA
	cudaSetDevice(compute_device_id);
	#endif
	
	// run the test
    if (argc == 4)
        run(size, threads);
    else run_file(threads, argv[4]);
    
    return 0;
}

