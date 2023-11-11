import importlib
import fl_prediction as model
import plotting as plot

importlib.reload(model)
loss_values = model.train("RNN")

plot.make_plot("RNN Loss per epoch", loss_values)
