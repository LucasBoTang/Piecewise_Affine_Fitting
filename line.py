import matplotlib.pyplot as plt

x = [50, 200, 600, 1200]
y1 = [0.1837, 0.1562, 0.1360, 0.1326]
y2 = [0.2968, 0.2576, 0.2326, 0.2087]
y3 = [0.6759, 0.6701, 0.4013, 0.3136]
y4 = [0.7402, 0.7260, 0.7052, 0.6853]

plt.plot(x, y1, "b*-", label="image 1 (600 pixels)")
plt.plot(x, y2, "g*-", label="image 2 (600 pixels)")
plt.plot(x, y3, "c*-", label="image 1 (2400 pixels)")
plt.plot(x, y4, "m*-", label="image 2 (2400 pixels)")

plt.ylim(0, 1)
plt.legend(loc=0)
plt.xlabel("Time Limit (seconds)")
plt.ylabel("Optimality Gap")

plt.show()
