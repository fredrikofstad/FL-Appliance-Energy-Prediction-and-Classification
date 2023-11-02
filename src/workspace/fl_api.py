from openfl.interface.interactive_api.experiment import FLExperiment
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface
from openfl.interface.interactive_api.federation import Federation

federation = Federation()

fl_experiment = FLExperiment(federation, "IN5460")
MI = ModelInterface(model, optimizer, framework_plugin: str)
