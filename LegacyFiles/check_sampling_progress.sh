#!/bin/bash

# Sampling Progress Monitor
# Checks the progress of large scale sampling

echo "=================================="
echo "scDiffusion Sampling Progress"
echo "Time: $(date)"
echo "=================================="

cd /home/suzuki/Projects/scDiffusion

# Check if process is still running
if pgrep -f "large_scale_sampling.sh" > /dev/null; then
    echo "✓ Sampling script is running"
elif pgrep -f "cell_sample.py\|classifier_sample.py\|batch_conditional_sample.py" > /dev/null; then
    echo "✓ Python sampling processes are running"
else
    echo "✗ No sampling processes detected"
fi

echo ""
echo "Generated files:"
echo "=================="

# Check unconditional samples
if [ -f "output/simulated_samples/pancreas/large_unconditional/unconditional_3000.npz" ]; then
    size=$(ls -lh "output/simulated_samples/pancreas/large_unconditional/unconditional_3000.npz" | awk '{print $5}')
    echo "✓ Unconditional (3000): ${size}"
else
    echo "⏳ Unconditional: In progress..."
fi

# Check conditional samples
echo ""
echo "Conditional samples (375 each):"
echo "--------------------------------"
conditional_count=0
for i in {0..7}; do
    file="output/simulated_samples/pancreas/large_conditional/conditional_type${i}.npz"
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✓ Cell type $i: ${size}"
        conditional_count=$((conditional_count + 1))
    else
        echo "⏳ Cell type $i: In progress..."
    fi
done

echo ""
echo "Progress summary:"
echo "=================="
echo "Conditional types completed: ${conditional_count}/8"

total_expected=$((3000 + 8 * 375))
echo "Expected total samples: ${total_expected}"

if [ -f "output/simulated_samples/pancreas/large_unconditional/unconditional_3000.npz" ]; then
    current_samples=$((3000 + conditional_count * 375))
    progress=$((current_samples * 100 / total_expected))
    echo "Current progress: ${current_samples}/${total_expected} samples (${progress}%)"
else
    current_samples=$((conditional_count * 375))
    progress=$((current_samples * 100 / total_expected))
    echo "Current progress: ${current_samples}/${total_expected} samples (${progress}%)"
fi

# Estimate time remaining (rough)
if [ $conditional_count -gt 0 ] && [ $conditional_count -lt 8 ]; then
    avg_time_per_type=11  # seconds based on observed performance
    remaining_types=$((8 - conditional_count))
    estimated_remaining=$((remaining_types * avg_time_per_type))
    echo "Estimated time remaining: ~${estimated_remaining} seconds"
fi

echo ""
