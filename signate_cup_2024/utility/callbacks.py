from imports import (
  optuna,
  Callback,
  NeuralNet,
  Any
)

class SkoPruneClb(Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()
        self._trial = trial
        self._monitor = monitor
        self.min_loss = float('inf')
    # def on_train_begin(self, net, X=None, y=None, **kwargs):
    #     """Called at the beginning of training."""

    # def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
    #     """Called at the beginning of each epoch."""

    def on_epoch_end(self, net: "NeuralNet", **kwargs: Any) -> None:
        history = net.history
        if not history:
            return
        epoch = len(history) - 1
        current_score = history[-1, self._monitor]
        self.min_loss = min(self.min_loss, current_score)
        self._trial.set_user_attr("min_loss", self.min_loss)
        self._trial.report(current_score, epoch)
        if self._trial.should_prune():
            message = f"///// Trial{self._trial.number} was pruned at epoch {epoch}. /////"
            raise optuna.TrialPruned(message)

# class StopWhenTrialKeepBeingPrunedCallback:
#     def __init__(self, threshold: int):
#         self.threshold = threshold
#         self._consequtive_pruned_count = 0

#     def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
#         if trial.state == optuna.trial.TrialState.PRUNED:
#             self._consequtive_pruned_count += 1
#         else:
#             self._consequtive_pruned_count = 0

#         if self._consequtive_pruned_count >= self.threshold:
#             print("///// study.stop() 発動 /////")
#             study.stop()