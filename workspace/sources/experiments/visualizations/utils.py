METRICS_PLOT_NAMES_MAPPING = {
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1-Score',
    'roc_auc': 'ROC-AUC',
    'loss': 'Loss',
    'accuracy': 'Accuracy',
    'false_positives_rate': 'FPR',
    'false_negatives_rate': 'FNR'
}


def pick_best_run_by_metric(runs, metric, second_metric=Loss):
    """Pick the best run based on given metric and its loss value."""
    metrics_keys = [f'test_{metric.name}_by_{metric.name}', f'test_{second_metric.name}_by_{metric.name}']
    metrics_greater_is_better = [metric.greater_is_better,
                                 second_metric.greater_is_better]
    metrics = zip(metrics_keys, metrics_greater_is_better)
    # Filter runs that have required metrics
    valid_runs = [run for run in runs
                  if all(key in run.data.metrics for key in metrics_keys)]

    if not valid_runs:
        return None

    # Define sort key function
    def _get_run_sort_key(run):
        return tuple(run.data.metrics[m_key] if m_gib else -run.data.metrics[m_key] for m_key, m_gib in metrics)

    # Return run with best metrics
    return max(valid_runs, key=_get_run_sort_key)


class InteractiveText:
    def __init__(self, text):
        self.text = text
        self.press = None
        self.selected = False
        self.cidpress = text.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = text.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = text.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidscroll = text.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cidkey = text.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def get_move_step(self):
        ax = self.text.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        step_x = (xlim[1] - xlim[0]) * 0.01  # 1% of axis width
        step_y = (ylim[1] - ylim[0]) * 0.01  # 1% of axis height
        return step_x, step_y

    def on_press(self, event):
        if event.inaxes != self.text.axes:
            return
        contains, _ = self.text.contains(event)
        if not contains:
            self.selected = False
            return
        self.press = (self.text.get_position(), (event.xdata, event.ydata))
        self.selected = True

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.text.axes:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.text.set_position((x0 + dx, y0 + dy))
        self.text.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.text.figure.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.text.axes:
            return
        contains, _ = self.text.contains(event)
        if not contains:
            return
        current_size = self.text.get_fontsize()
        new_size = max(1, current_size + event.step)
        self.text.set_fontsize(new_size)
        self.text.figure.canvas.draw_idle()

    def on_key(self, event):
        if not self.selected:
            return
        x, y = self.text.get_position()
        step_x, step_y = self.get_move_step()
        if event.key == 'backspace':
            self.text.set_text(self.text.get_text()[:-1])
        elif event.key == 'left':
            self.text.set_position((x - step_x, y))
        elif event.key == 'right':
            self.text.set_position((x + step_x, y))
        elif event.key == 'up':
            self.text.set_position((x, y + step_y))
        elif event.key == 'down':
            self.text.set_position((x, y - step_y))
        elif len(event.key) == 1:
            self.text.set_text(self.text.get_text() + event.key)
        self.text.figure.canvas.draw_idle()
