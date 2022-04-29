import pytorch_lightning as pl


class TeacherForcingAnnealCallback(pl.Callback):
    def __init__(self, start, end, steps):
        assert start >= end
        self.start = start
        self.end = end
        self.steps = steps
        self.slope = float((start - end)) / steps

    def on_train_batch_end(self, trainer, model, *args, **kwargs):
        current = model.teacher_forcing_prob
        new_teacher_forcing_prob = max(self.end, current - self.slope)
        model.teacher_forcing_prob = new_teacher_forcing_prob
        model.log("teacher_forcing_prob", new_teacher_forcing_prob)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--teacher_forcing_start", type=float, default=0.8)
        parser.add_argument("--teacher_forcing_end", type=float, default=0.0)
        parser.add_argument("--teacher_forcing_anneal_steps", type=int, default=8000)


class TimeMaskedLossCallback(pl.Callback):
    def __init__(self, start, end, steps):
        assert start <= end
        self.start = start
        self.end = end
        self.steps = steps
        self.slope = float((end - start)) / steps
        self._time_mask = self.start

    @property
    def time_mask(self):
        return round(self._time_mask)

    def on_train_start(self, trainer, model):
        if model.time_masked_idx is None:
            model.time_masked_idx = self.time_mask

    def on_train_batch_end(self, trainer, model, *args):
        self._time_mask = min(self.end, self._time_mask + self.slope)
        model.time_masked_idx = self.time_mask
        model.log("time_masked_idx", self.time_mask)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--time_mask_start", type=int, default=1)
        parser.add_argument("--time_mask_end", type=int, default=12)
        parser.add_argument("--time_mask_anneal_steps", type=int, default=1000)
        parser.add_argument("--time_mask_loss", action="store_true")
