from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_train = False
        self._parser.add_argument('--names',default =[], type=str, nargs="+",
                                    help='A list of experiment names which are stored under checkpoints directory can form a deep ensemble.')
        self._parser.add_argument('--load_epochs', type=int, default=[],nargs="+",
                                    help='the load epochs for each experiment.')

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        assert len(self._opt.names) == len(self._opt.load_epochs), "The number of names should be the same as the number of load epochs"
        # set is train or test
        self._opt.is_train = self.is_train

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        return self._opt