class Annotator:
    def __init__(self, data):
        self.data = data

    def provide_annotation_interface(self):
        """Provides a simple annotation interface using Matplotlib."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data)
        plt.title("Annotate Data")
        plt.xlabel("Time")
        plt.ylabel("Value")

        def on_click(event):
            if event.inaxes:
                x, y = event.xdata, event.ydata
                print(f"Clicked at: ({x}, {y})")
                # Add annotation logic here (e.g., store annotation coordinates)

        cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show()