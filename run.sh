#!/usr/bin/env bash

#CONNECTION_STRING="postgresql://ubuntu@localhost:5432"
CONNECTION_STRING="postgresql://tjg@localhost:28817/vectorscale"

# BUGGY CASE 1: 2 labels, 1M scale => recall is 0.4920.  Works fine at smaller scales.
cargo run --release --bin bench -- \
    --connection-string $CONNECTION_STRING \
    test \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --table cohere_wikipedia_22_12_1m_angular_uniform_diskann_filt \
    --ground-truth-table cohere_wikipedia_22_12_1m_angular_uniform \
    --max-label 32 \
    --num-labels 2 \
    --num-queries 100

# BUGGY CASE 2: 2 labels, 1k scale, post-filtering with diskann => recall is 0.7570.  Works fine at smaller scales.
cargo run --release --bin bench -- \
    --connection-string $CONNECTION_STRING \
    test \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --table cohere_wikipedia_22_12_1k_angular_uniform_diskann \
    --ground-truth-table cohere_wikipedia_22_12_1k_angular_uniform \
    --max-label 32 \
    --num-labels 2 \
    --num-queries 100

cargo run --release --bin bench -- \
    --connection-string $CONNECTION_STRING \
    test \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --table cohere_wikipedia_22_12_1k_angular_uniform_diskann \
    --ground-truth-table cohere_wikipedia_22_12_1k_angular_uniform \
    --max-label 32 \
    --num-labels 2 \
    --num-queries 100 \
    --diskann-query-search-list-size 400

# BUGGY CASE 3: no labels, 1k scale => recall is 0.6140.  Works fine at smaller scales.
cargo run --release --bin bench -- \
    --connection-string $CONNECTION_STRING \
    test \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --table cohere_wikipedia_22_12_1k_angular_uniform_diskann_filt \
    --ground-truth-table cohere_wikipedia_22_12_1k_angular_uniform \
    --num-queries 100

cargo run --release --bin bench -- \
    --connection-string $CONNECTION_STRING \
    load \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --table cohere_wikipedia_22_12_1k_angular_uniform \
    --create-table \
    --max-label 32 \
    --num-labels 1 \
    --num-vectors 1000








# load 100
./target/release/bench \
    --connection-string $CONNECTION_STRING \
    load \
    --create-table \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --max-label 4 \
    --num-labels 1 \
    --table cohere_wikipedia_22_12_100_angular_uniform_diskann_filt \
    --num-vectors 100

./target/release/bench \
    --connection-string $CONNECTION_STRING \
    test \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --table cohere_wikipedia_22_12_1k_angular_uniform_diskann_filt \
    --ground-truth-table cohere_wikipedia_22_12_1k_angular_uniform \
    --max-label 32 \
    --num-labels 1 \
    --num-queries 100


# load 1000
cargo run --release --bin bench -- \
    --connection-string $CONNECTION_STRING \
    load \
    --create-table \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --max-label 32 \
    --num-labels 1 \
    --table cohere_wikipedia_22_12_1k_angular_uniform_diskann_filt \
    --num-vectors 1000

./target/release/bench \
    --connection-string $CONNECTION_STRING \
    test \
    --dataset cohere-wikipedia-22-12-1M-angular \
    --table cohere_wikipedia_22_12_1k_angular_uniform_diskann_filt \
    --ground-truth-table cohere_wikipedia_22_12_1k_angular_uniform \
    --max-label 32 \
    --num-labels 1 \
    --num-queries 100

for table in \
    cohere_wikipedia_22_12_1m_angular_uniform_diskann_filt \
    cohere_wikipedia_22_12_1m_angular_uniform \
    cohere_wikipedia_22_12_1m_angular_uniform_int \
    cohere_wikipedia_22_12_1m_angular_uniform_diskann \
    cohere_wikipedia_22_12_1m_angular_normal_diskann_filt \
    cohere_wikipedia_22_12_1m_angular_normal \
    cohere_wikipedia_22_12_1m_angular_normal_int \
    cohere_wikipedia_22_12_1m_angular_normal_diskann
do
    for i in {1..3}
    do
        for num_labels in 1 2 4 8 16 32 64
        do
            echo "----------------------------------------"
            echo "$table run $i num_labels $num_labels"
            echo "----------------------------------------"
            ./target/release/bench \
                --connection-string $CONNECTION_STRING \
                test \
                --dataset cohere-wikipedia-22-12-1M-angular \
                --max-label 64 \
                --num-labels $num_labels \
                --table $table \
                --num-queries 1000
            echo "----------------------------------------"
        done
    done
done