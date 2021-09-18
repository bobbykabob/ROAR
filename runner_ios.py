from pathlib import Path
from ROAR_iOS.ios_runner import iOSRunner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_iOS.config_model import iOSConfig
# from ROAR.agent_module.ios_agent import iOSAgent
from ROAR.agent_module.free_space_auto_agent import FreeSpaceAutoAgent
# from ROAR.agent_module.line_following_agent_2 import LineFollowingAgent
from ROAR.utilities_module.vehicle_models import Vehicle
import logging
import argparse
from misc.utils import str2bool


class mode_list(list):
    # list subclass that uses lower() when testing for 'in'
    def __contains__(self, other):
        return super(mode_list, self).__contains__(other.lower())


if __name__ == '__main__':
    choices = mode_list(['ar', 'vr'])
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", type=str2bool, default=False, help="True to use auto control")
    parser.add_argument("-m", "--mode", choices=choices, help="AR or VR", default="vr")
    args = parser.parse_args()

    try:
        agent_config = AgentConfig.parse_file(
            Path("ROAR/configurations/iOS/iOS_agent_configuration.json")
        )
        ios_config = iOSConfig.parse_file(
            Path("ROAR_iOS/configurations/ios_config.json")
        )
        ios_config.ar_mode = True if args.mode == "ar" else False

        agent = FreeSpaceAutoAgent(vehicle=Vehicle(), agent_settings=agent_config, should_init_default_cam=True)
        ios_runner = iOSRunner(agent=agent, ios_config=ios_config)
        ios_runner.start_game_loop(auto_pilot=args.auto)
    except Exception as e:
        print(f"Something bad happened: {e}")
