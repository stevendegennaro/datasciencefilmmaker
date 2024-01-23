import matplotlib.pyplot as plt
import numpy as np

keras_argmax = [59.0, 43.5]
keras_sample_from = [48.0, 30.3]

scratch_argmax = [56.9,40.0]
scratch_sample_from = [44.8,26.8]

theoretical_argmax = [62.7, 60.1]
theoretical_sample_from = [54.1, 53.6]

which = 0

method = ("sample_from", "argmax")
accuracy = {
    'Scratch Network': (scratch_sample_from[which], scratch_argmax[which]),
    'Keras Network': (keras_sample_from[which], keras_argmax[which]),
    'Theoretical Max': (theoretical_sample_from[which], theoretical_argmax[which]),
}

x = np.arange(len(method))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in accuracy.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Accuracy Comparison for {"Last" if which else "First"} Names')
ax.set_xticks(x + width, method)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)

plt.show()