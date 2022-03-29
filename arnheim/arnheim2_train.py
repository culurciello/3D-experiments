# E. Culurciello
# December 2021

# TURN YOUR PICTURES INTO ART!
# arnheim 2 adaptation and training
# https://colab.research.google.com/github/deepmind/arnheim/blob/master/arnheim_2.ipynb

import os
import PIL.Image
import random
import time
from argparse import ArgumentParser

from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import subprocess
import clip
import pydiffvg

os.environ["FFMPEG_BINARY"] = "ffmpeg"
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

title = 'Arnheim 2 - Synthesize drawings to match a text prompt'

parser = ArgumentParser(description=title)
arg = parser.add_argument
arg('-i', type=str, default='an abandoned plane on a field', help='text prompt')
# arg('--canvas_dim', type=int, default=224, help='cavas size')
# arg('--num_paths', type=int, default=256, help='number of paths')
# arg('--num_iter', type=int, default=1000, help='number of training iterations')
# arg('--max_width', type=int, default=50, help='max width')
# arg('--gamma', type=float, default=1.0, help='gama')
arg('--seed', type=int, default=789, help='random seed')
arg('--cuda', default=False, action='store_true', help='use GPU or CPU?')
arg('--device_num', type=int, default=0, help='GPU number')
arg('--workers', type=int, default=8, help='number of CPU workers / threads')
args = parser.parse_args()

# Setup
# random seeds and reproducible results:
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)
print("Torch version:", torch.__version__)

if torch.cuda.is_available():# and args.cuda:
    args.use_gpu = 1
    device = torch.device("cuda:"+str(args.device_num))
    torch.cuda.set_device(device)
    print('Using CUDA!')
else:
    args.use_gpu = 0
    device = torch.device("cpu")
    print('Using CPU')


# Configure pydiffvg and load CLIP model
pydiffvg.set_print_timing(False)
pydiffvg.set_device(device)
pydiffvg.set_use_gpu(torch.cuda.is_available())  # Use GPU if available.

CLIP_MODEL = "ViT-B/32"
print('Using CLIP model', CLIP_MODEL)
clip_model, _ = clip.load(CLIP_MODEL, device, jit=False)


# Grammar Drawing Network Definitions

# Grammar Specific Drawing Network: Photographic

class CurveNetworkPhotographicLSTM(torch.nn.Module):
  """LSTM-based line-properties with CLIPDraw-like line optimization."""

  def __init__(self):
    """Constructor, relying on global parameters."""
    super().__init__()

    # There are 3 LSTMS, for points, widths and colours.
    assert NUM_LSTMS == 3, "Need exactly 3 LSTMs."

    # Bézier curve parameterisations are added to the generative grammar.
    points = []
    for _ in range(BATCH_SIZE * NUM_STROKE_TYPES * SEQ_LENGTH):
      p0 = (1.5 * (random.random() - 0.5), 1.5 * (random.random() - 0.5))
      points.append(p0)
      radius = 0.1
      p1 = (p0[0] + radius * (
          random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
      p2 = (p1[0] + radius * (
          random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
      p3 = (p2[0] + radius * (
          random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
      points.append(p1)
      points.append(p2)
      points.append(p3)
      p0 = p3
    points = np.array(points).flatten()
    self.positions = torch.nn.Parameter(torch.Tensor(points))

    # Initial sequences.
    self._initials = []
    for _ in range(NUM_LSTMS):
      initial = torch.nn.Parameter(torch.ones(
          BATCH_SIZE, SEQ_LENGTH, INPUT_SPEC_SIZE))
      self._initials.append(initial)

    # Shared input layer to process the initial sequence.
    self._input_layer = torch.nn.Sequential(
        torch.nn.Linear(INPUT_SPEC_SIZE, NET_LSTM_HIDDENS),
        torch.nn.LeakyReLU(0.2, inplace=True))

    # Different initial sequences, LSTMs and heads for lines, colorus, width.
    lstms = []
    heads = []
    for _ in range(NUM_LSTMS):
      lstm_layer = torch.nn.LSTM(
          input_size=NET_LSTM_HIDDENS, hidden_size=NET_LSTM_HIDDENS,
          num_layers=1, batch_first=True, bias=True)
      head_layer = torch.nn.Sequential(
          torch.nn.Linear(NET_LSTM_HIDDENS, NET_MLP_HIDDENS),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Linear(NET_MLP_HIDDENS, OUTPUT_SIZE))
      lstms.append(lstm_layer)
      heads.append(head_layer)
    self._lstms = torch.nn.ModuleList(lstms)
    self._heads = torch.nn.ModuleList(heads)

  def forward(self):
    """Input-less forward function."""
    pred = []
    for i in range(NUM_LSTMS):
      x = self._input_layer(self._initials[i])
      y, _ = self._lstms[i](x)
      y = torch.reshape(self._heads[i](y), (BATCH_SIZE*SEQ_LENGTH*OUTPUT_SIZE,))
      if i == 0:
        y = y * OUTPUT_COEFF_SYSTEMATICITY + self.positions
        points = torch.clamp(y, min=-1, max=1)
        pred.append(points)
      elif i == 1:
        widths = torch.clamp(y * OUTPUT_COEFF_WIDTH, min=1, max=100)
        pred.append(widths)
      elif i == 2:
        colours = torch.clamp(y * OUTPUT_COEFF_COLOUR, min=0, max=1)
        pred.append(colours)
    # Unused diversity loss term.
    pred.append(0)
    return pred

#@title Grammar Specific Drawing Network: Arnheim 2

class CurveNetworkHierarchicalLSTM(torch.nn.Module):
  """LSTM-based production of drawing sequences."""

  def __init__(self):
    super().__init__()

    def init_all(model, init_func, *params, **kwargs):
      """Init all parameters of model.

      Args:
        model: TF model top initialise
        init_func: TF initialisation function
        *params: Params to pass to init_func
        **kwargs: kwargs to pass to init_func

      Returns:
        None
      """
      for p in model.parameters():
        init_func(p, *params, **kwargs)

    # ********************* Direct encoding*************************************
    points_in = []
    for _ in range(BATCH_SIZE * LSTM_ITERATIONS * LSTM_ITERATIONS):
      p0 = (1.5 * (random.random()-0.5), 1.5 * (random.random() - 0.5))
      points_in.append(p0)
      radius = 0.1
      p1 = (p0[0] + radius * (
          random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
      p2 = (p1[0] + radius * (
          random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
      p3 = (p2[0] + radius * (
          random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
      points_in.append(p1)
      points_in.append(p2)
      points_in.append(p3)
      p0 = p3

    self.direct_points = torch.nn.Parameter(
        torch.Tensor(np.array(points_in).flatten().reshape(
            (BATCH_SIZE*LSTM_ITERATIONS*LSTM_ITERATIONS, 4, 2))))

    # Set values to torch.nn.Parameters to make them evolvable.
    self.positions_top = torch.nn.Parameter(
        2.0 * (torch.rand(BATCH_SIZE, EMBEDDING_SIZE) - 0.5))
    # Direct vs indirect position weighting
    # One can experiment with replacing the constants below with learnable
    # parameters, see suggestions in the comments following them.
    self.scale_a = 0.01  # torch.nn.Parameter(1.0*(torch.rand(1)))
    # Bottom-level position scaling
    self.scale_b = 4  # torch.nn.Parameter(4.0*(torch.rand(1)))
    # Softmax logit scaling
    self.scale_c = 1  # torch.nn.Parameter(1.0*(torch.rand(1)))
    # Point scaling
    self.scale_d = 3  # torch.nn.Parameter(10.0*(torch.rand(1)))
    # Width scaling
    self.scale_e = 2  # torch.nn.Parameter(1.0*(torch.rand(1)))
    # Colour scaling
    self.scale_f = 1.5  # torch.nn.Parameter(3.0*(torch.rand(1)))
    if LEARNABLE_FORCING:
      self.scale_g = torch.nn.Parameter(1.0 * (torch.rand(1)))
    if VERBOSE:
      print("self.positions_top", self.positions_top.shape)

    lstms_top = []
    heads_top = []
    for _ in range(NUM_LSTMS):
      lstm_layer_top = torch.nn.LSTM(
          input_size=EMBEDDING_SIZE, hidden_size=NET_LSTM_HIDDENS,
          num_layers=2, batch_first=True, bias=True)
      if WEIGHT_INITIALIZER:
        init_all(lstm_layer_top, torch.nn.init.normal_, mean=0., std=WEIGHT_STD)

      head_layer_top = torch.nn.Sequential(
          torch.nn.Linear(NET_LSTM_HIDDENS, NET_MLP_HIDDENS),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Linear(NET_MLP_HIDDENS, OUTPUT_SIZE))

      lstms_top.append(lstm_layer_top)
      heads_top.append(head_layer_top)
    self._lstms_top = torch.nn.ModuleList(lstms_top)
    self._heads_top = torch.nn.ModuleList(heads_top)

    lstms = []
    heads = []
    for _ in range(NUM_LSTMS):

      lstm_layer = torch.nn.LSTM(
          input_size=EMBEDDING_SIZE, hidden_size=NET_LSTM_HIDDENS,
          num_layers=2, batch_first=True, bias=True)

      if WEIGHT_INITIALIZER:
        init_all(lstm_layer, torch.nn.init.normal_, mean=0., std=WEIGHT_STD)

      head_layer = torch.nn.Sequential(
          torch.nn.Linear(NET_LSTM_HIDDENS, NET_MLP_HIDDENS),
          torch.nn.LeakyReLU(0.2, inplace=True),
          torch.nn.Linear(NET_MLP_HIDDENS, OUTPUT_SIZE))
      lstms.append(lstm_layer)
      heads.append(head_layer)
    self._lstms = torch.nn.ModuleList(lstms)
    self._heads = torch.nn.ModuleList(heads)

  def forward(self):
    # TOP LEVEL
    self.interleaved_positions_top = self.positions_top[:, :]
    self.interleaved_positions_top = self.interleaved_positions_top.unsqueeze(
        1).repeat(1, LSTM_ITERATIONS, 1)
    if VERBOSE:
      print("self.interleaved_positions_top",
            self.interleaved_positions_top.shape)

    self.softmax_logits_top = self.positions_top[:, 2:2 + NUM_LSTMS]

    if VERBOSE:
      print("self.softmax_logits_top", self.softmax_logits_top.shape)

    self.non_parametric_positions_top = self.positions_top[:, :2]

    if VERBOSE:
      print("self.non_parametric_positions_top",
            self.non_parametric_positions_top.shape)

    self.non_parametric_positions_repeated_top = (
        self.non_parametric_positions_top.unsqueeze(1).repeat(
            1, LSTM_ITERATIONS, 1).reshape(
                BATCH_SIZE*LSTM_ITERATIONS, 2))

    if VERBOSE:
      print("self.non_parametric_positions_repeated_top",
            self.non_parametric_positions_repeated_top.shape)

    self.softmax_outputs_top = torch.nn.functional.normalize(
        self.softmax_logits_top, p=2, dim=1)

    #TOP LEVEL NETWORK FORWARD PASS
    pred_top = []
    for i in range(NUM_LSTMS):
      x_top = self.interleaved_positions_top*10.0
      if USE_DROPOUT:
        y_top, _ = self._lstms_top[i](
            torch.nn.Dropout(DROPOUT_PROP)(x_top))
        y_top = self._heads_top[i](
            torch.nn.Dropout(DROPOUT_PROP)(y_top))
      else:
        y_top, _ = self._lstms_top[i](x_top)
        y_top = self._heads_top[i](y_top)
      pred_top.append(y_top)
    preds_top = torch.stack(pred_top, axis=1)
    out_top = torch.einsum("bijk,bi->bjk", preds_top, self.softmax_outputs_top)
    out_top = out_top.reshape(BATCH_SIZE*LSTM_ITERATIONS, EMBEDDING_SIZE)

    # TOGGLE CRASHES TO INCLUDE THE TOP LEVEL NETWORK IN THE GRAPH.
    crashes = True
    if not crashes:
      input_layer = self.input
    else:
      input_layer = out_top

    # BOTTOM LEVEL
    if VERBOSE:
      print("BOTTOM LEVEL <<<<<<<<<<<<")

    self.non_parametric_positions = (self.non_parametric_positions_repeated_top
                                     + self.scale_a * input_layer[:, :2])
    if VERBOSE:
      print("self.non_parametric_positions",
            self.non_parametric_positions.shape)

    self.non_parametric_positions_repeated = (
        self.non_parametric_positions.unsqueeze(1).repeat(
            1, LSTM_ITERATIONS, 1).reshape(
                (BATCH_SIZE * LSTM_ITERATIONS * LSTM_ITERATIONS, 2)))
    if VERBOSE:
      print("self.non_parametric_positions_repeated",
            self.non_parametric_positions_repeated.shape)
    interleaved_positions = self.scale_b * input_layer[:, :]

    interleaved_positions = interleaved_positions.unsqueeze(1).repeat(
        1, LSTM_ITERATIONS, 1)
    if VERBOSE:
      print("interleaved_positions", interleaved_positions.shape)
    softmax_logits = self.scale_c * input_layer[:, 2:2 + NUM_LSTMS]
    softmax_outputs = torch.nn.functional.normalize(softmax_logits, p=2, dim=1)

    pred = []
    for i in range(NUM_LSTMS):
      x = interleaved_positions*10.0
      if USE_DROPOUT:
        y, _ = self._lstms[i](torch.nn.Dropout(DROPOUT_PROP)(x))
        y = self._heads[i](torch.nn.Dropout(DROPOUT_PROP)(y))
      else:
        y, _ = self._lstms[i](x)
        y = self._heads[i](y)

      pred.append(y)
    preds = torch.stack(pred, axis=1)
    out = torch.einsum("bijk,bi->bjk", preds, softmax_outputs)
    out = out.reshape(
        (BATCH_SIZE * LSTM_ITERATIONS * LSTM_ITERATIONS, EMBEDDING_SIZE))

    self.non_parametric_positions_repeated = (
        self.non_parametric_positions_repeated.repeat(1, 4))
    if VERBOSE:
      print("self.non_parametric_positions_repeated",
            self.non_parametric_positions_repeated.shape)

    if VERBOSE:
      print("out shape", out.shape)

    if YELLOW_ABSOLUTE_POSITIONS_USED:
      points = (out[:, :8] * self.scale_d
                + self.non_parametric_positions_repeated)
    else:
      points = (out[:, :8] * self.scale_d
                + self.non_parametric_positions_repeated * 0.0)

    points = points.reshape(
        (BATCH_SIZE*LSTM_ITERATIONS * LSTM_ITERATIONS, 4, 2))
    if LEARNABLE_FORCING:
      points = (points * self.scale_g
                + torch.nn.self.direct_points * (1-self.scale_g))
    else:
      points = (points * GRAMMATICAL_FORCING
                + self.direct_points * (1-GRAMMATICAL_FORCING))

    widths = out[:, 8] * self.scale_e * 20.0
    colours = out[:, 9:13] * self.scale_f * 6.0

    points = torch.clamp(points, min=-1.1, max=1.1)
    widths = torch.clamp(widths, min=1, max=20)
    colours = torch.clamp(colours, min=0.0, max=1)

    return points, widths, colours, 0

#@title Grammar Specific Drawing Network: DPPN Generative Grammar

class CurveNetworkDPPN(torch.nn.Module):
  """DPPN-based production of drawing sequences."""

  def __init__(self):
    super().__init__()

    def init_all(model, init_func, *params, **kwargs):
      for p in model.parameters():
        init_func(p, *params, **kwargs)

    # ***************************Direct encoding**********************
    points_in = []
    for _ in range(BATCH_SIZE):
      p0 = (1.5*(random.random() - 0.5), 1.5 * (random.random() - 0.5))
      points_in.append(p0)
      radius = 0.1
      p1 = (p0[0] + radius * (random.random() - 0.5),
            p0[1] + radius * (random.random() - 0.5))
      p2 = (p1[0] + radius * (random.random() - 0.5),
            p1[1] + radius * (random.random() - 0.5))
      p3 = (p2[0] + radius * (random.random() - 0.5),
            p2[1] + radius * (random.random() - 0.5))
      points_in.append(p1)
      points_in.append(p2)
      points_in.append(p3)
      p0 = p3

    self.direct_points = torch.nn.Parameter(torch.Tensor(
        np.array(points_in).flatten().reshape((BATCH_SIZE, 4, 2))))

    # ****************************Indirect encoding********************

    self.positions_top = torch.nn.Parameter(
        2.0 * (torch.rand(BATCH_SIZE, EMBEDDING_SIZE) - 0.5))

    self.scale_d = 0.2  #torch.nn.Parameter(10.0*(torch.rand(1))) #Point scaling
    self.scale_e = 4  #torch.nn.Parameter(1.0*(torch.rand(1))) #Width scaling
    self.scale_f = 1  #torch.nn.Parameter(3.0*(torch.rand(1))) #Colour scaling
    if LEARNABLE_FORCING:
      self.scale_g = torch.nn.Parameter(1.0*(torch.rand(1)))
    if VERBOSE:
      print("self.positions_top", self.positions_top.shape)

    self.ffnn = torch.nn.Sequential(
        torch.nn.Linear(EMBEDDING_SIZE, NET_MLP_HIDDENS),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(NET_MLP_HIDDENS, NET_MLP_HIDDENS),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(NET_MLP_HIDDENS, NET_MLP_HIDDENS),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(NET_MLP_HIDDENS, OUTPUT_SIZE))

    if WEIGHT_INITIALIZER:
      init_all(self.ffnn, torch.nn.init.normal_, mean=0., std=WEIGHT_STD)

  def forward(self):
    # TOP LEVEL
    self.non_parametric_positions_top = self.positions_top[:, :2]
    out = self.ffnn(self.positions_top)
    self.non_parametric_positions_repeated = (
        self.non_parametric_positions_top.repeat(1, 4))
    points = (out[:, :8] * self.scale_d
              + self.non_parametric_positions_repeated)

    points = points.reshape((BATCH_SIZE, 4, 2))
    if LEARNABLE_FORCING:
      points = (points * self.scale_g
                + torch.nn.self.direct_points * (1 - self.scale_g))
    else:
      points = (points * GRAMMATICAL_FORCING
                + self.direct_points * (1 - GRAMMATICAL_FORCING))

    widths = out[:, 8] * self.scale_e
    colours = out[:, 9:13] * self.scale_f

    points = torch.clamp(points, min=-1.1, max=1.1)
    widths = torch.clamp(widths, min=1, max=20)
    colours = torch.clamp(colours, min=0.0, max=1)

    return points, widths, colours, 0

#@title Grammar Specific Drawing Network: SEQ-TO-SEQ
class EncoderRNN(torch.nn.Module):
  """RNN-based sequence encoder."""

  def __init__(self, input_size, hidden_size):
    """Constructor.

    Args:
      input_size: number of input units
      hidden_size: number of hidden units
    """
    super(EncoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        batch_first=True,
        bias=True)

  def forward(self, input_layer, hidden_layer):

    output_layer, hidden_layer = self.gru(input_layer, hidden_layer)
    return output_layer, hidden_layer

  def init_hidden(self):
    return torch.zeros(2, BATCH_SIZE, self.hidden_size)


class DecoderRNN(torch.nn.Module):
  """RNN-based sequence decoder."""

  def __init__(self, hidden_size, output_size):
    """Constructor.

    Args:
      hidden_size: number of hidden units
      output_size: number of output units
    """
    super(DecoderRNN, self).__init__()
    self.hidden_size = hidden_size

    self.embedding = torch.nn.Embedding(output_size, hidden_size)
    self.gru = torch.nn.GRU(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=2,
        batch_first=True,
        bias=True)
    self.out = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.LeakyReLU(0.2, inplace=True),
        torch.nn.Linear(hidden_size, output_size))

  def forward(self, input_layer, hidden):
    output = self.embedding(input_layer)
    output = F.relu(output)
    output, hidden = self.gru(output, hidden)
    output = self.out(output)
    return output, hidden


class CurveNetworkSEQTOSEQ(torch.nn.Module):
  """Sequence-to-sequence-based production of drawing sequences."""

  def __init__(self):
    """Constructor, relying on global parameters."""
    super().__init__()

    def init_all(model, init_func, *params, **kwargs):
      for p in model.parameters():
        init_func(p, *params, **kwargs)

    self.hidden_size = NET_LSTM_HIDDENS
    self.input_length = SEQ_LENGTH
    self.output_length = NUM_STROKES
    self.input_size = INPUT_SIZE
    self.output_size = OUTPUT_SIZE

    # *************************************Direct encoding*************************************
    points_in = []
    for _ in range(self.output_length * BATCH_SIZE):
      p0 = (1.5 * (random.random() - 0.5), 1.5 * (random.random() - 0.5))
      points_in.append(p0)
      radius = 0.1
      p1 = (p0[0] + radius * (random.random() - 0.5),
            p0[1] + radius * (random.random() - 0.5))
      p2 = (p1[0] + radius * (random.random() - 0.5),
            p1[1] + radius * (random.random() - 0.5))
      p3 = (p2[0] + radius * (random.random() - 0.5),
            p2[1] + radius * (random.random() - 0.5))
      points_in.append(p1)
      points_in.append(p2)
      points_in.append(p3)
      p0 = p3

    self.direct_points = torch.nn.Parameter(
        torch.Tensor(
            np.array(points_in).flatten().reshape(
                (self.output_length * BATCH_SIZE, 4, 2))))

    #input_tensor
    self.positions_top = torch.nn.Parameter(
        2.0 * (torch.rand(BATCH_SIZE, SEQ_LENGTH, self.input_size) - 0.5))
    self.positions_top2 = torch.nn.Parameter(2.0 *
                                             (torch.rand(BATCH_SIZE, 2) - 0.5))

    self.scale_d = 0.4  #torch.nn.Parameter(10.0*(torch.rand(1))) #Point scaling
    self.scale_e = 3.0  #torch.nn.Parameter(1.0*(torch.rand(1))) #Width scaling
    self.scale_f = 1.0  #torch.nn.Parameter(3.0*(torch.rand(1))) #Colour scaling

    if LEARNABLE_FORCING:
      self.scale_g = torch.nn.Parameter(1.0 * (torch.rand(1)))
    if VERBOSE:
      print("self.positions_top", self.positions_top.shape)

    self.encoder1 = EncoderRNN(self.input_size, self.hidden_size)
    self.decoder1 = DecoderRNN(self.hidden_size, self.output_size)
    if WEIGHT_INITIALIZER:
      init_all(self.encoder1, torch.nn.init.normal_, mean=0., std=WEIGHT_STD)
      init_all(self.decoder1, torch.nn.init.normal_, mean=0., std=WEIGHT_STD)

    self.encoder_outputs = torch.zeros(self.output_length,
                                       self.encoder1.hidden_size)
    self.decoder_input = torch.tensor(np.zeros((BATCH_SIZE, 1), dtype=int))

  def forward(self):

    self.non_parametric_positions_repeated = (
        self.positions_top2.unsqueeze(1).repeat(
            1, self.output_length, 1).reshape(BATCH_SIZE, self.output_length,
                                              2))
    self.non_parametric_positions_repeated = (
        self.non_parametric_positions_repeated.repeat(1, 1, 4).reshape(
            (BATCH_SIZE * self.output_length, 8)))

    self.encoder_hidden = self.encoder1.init_hidden()

    # Input-less forward function.
    self.encoder_output, self.encoder_hidden = self.encoder1(
        self.positions_top * 3.0, self.encoder_hidden)

    decoder_hidden = self.encoder_hidden
    decoder_input = self.decoder_input
    output_history = []

    # Without teacher forcing: use its own predictions as the next input
    for _ in range(self.output_length):

      decoder_output, decoder_hidden = self.decoder1(decoder_input,
                                                     decoder_hidden)

      _, topi = decoder_output.topk(1)
      # detach from history as input
      decoder_input = topi.squeeze().detach().reshape((BATCH_SIZE, 1))
      output_history.append(decoder_output)

    out = torch.stack(output_history, axis=0)
    out = torch.reshape(out,
                        (BATCH_SIZE * self.output_length, self.output_size))

    if VERBOSE:
      print("out shape", out.shape)
    if YELLOW_ABSOLUTE_POSITIONS_USED:

      points = (
          out[:, :8] * self.scale_d + self.non_parametric_positions_repeated)
    else:
      points = out[:, :8] * self.scale_d

    points = points.reshape(self.output_length * BATCH_SIZE, 4, 2)
    if LEARNABLE_FORCING:
      points = (
          points * (self.scale_g) + self.direct_points * (1 - self.scale_g))
    else:
      points = (
          points * (GRAMMATICAL_FORCING) + self.direct_points *
          (1 - GRAMMATICAL_FORCING))

    widths = out[:, 8] * self.scale_e
    colours = out[:, 9:13] * self.scale_f

    points = torch.clamp(points, min=-1.1, max=1.1)
    widths = torch.clamp(widths, min=1, max=20)
    colours = torch.clamp(colours, min=0.0, max=1)

    return points, widths, colours, 0

"""# Function definitions"""

#@title Image rendering and display

def show_and_save(img, t=None, dpi=75, figsize=(5, 5)):
  """Display image.

  Args:
    img: image to display
    t: time step
    dpi: display resolution
    figsize: size of image
  """
  _ = plt.figure(figsize=figsize, dpi=dpi)
  img = np.transpose(img.detach().cpu().numpy()[0], (1, 2, 0))
  img = np.clip(img, 0.0, 1.0)
  plt.imshow(img, interpolation="None")
  plt.grid(None)
  plt.axis("off")
  if img.shape[1] > CANVAS_WIDTH:
    type_image = "highres_image_"
  else:
    type_image = "image_"
  path_fig = DIR_RESULTS + "/" + type_image + PROMPT
  if t is not None:
    path_fig += "_t_" + str(t)
  path_fig += ".png"
  plt.savefig(path_fig, dpi=dpi)
  # plt.show()


def render(all_points, all_widths, all_colors, multiplier=1):
  """Render line date to image.

  Args:
    all_points: points defining the lines
    all_widths: the widths of the lines
    all_colors: line colours
    multiplier: scale factor to enlarge drawing

  Returns:
    image with lines drawn
  """
  # Store `all_points` definitions as shapes, colours and widths.
  shapes = []
  shape_groups = []
  for p in range(all_points.shape[0]):
    points = all_points[p].contiguous().cpu()
    width = all_widths[p].cpu()
    color = all_colors[p].cpu()
    num_ctrl_pts = torch.zeros(NUM_SEGMENTS, dtype=torch.int32) + 2
    path = pydiffvg.Path(
        num_control_points=num_ctrl_pts, points=points * multiplier,
        stroke_width=width * multiplier, is_closed=False)
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([len(shapes) - 1]),
        fill_color=None,
        stroke_color=color)
    shape_groups.append(path_group)

  # Rasterize the image.
  scene_args = pydiffvg.RenderFunction.serialize_scene(
      CANVAS_WIDTH * multiplier,
      CANVAS_HEIGHT * multiplier,
      shapes, shape_groups)
  img = pydiffvg.RenderFunction.apply(
      CANVAS_WIDTH * multiplier,
      CANVAS_HEIGHT * multiplier,
      2, 2, 0, None, *scene_args)
  if DRAW_WHITE_BACKGROUND:
    w, h = img.shape[0], img.shape[1]
    img = img[:, :, 3:4] * img[:, :, :3] + (
        torch.ones(w, h, 3, device=pydiffvg.get_device()) * (1-img[:, :, 3:4]))
  else:
    img = img[:, :, :3]
  img = img.unsqueeze(0)

  return img

#@title Image augmentation transformations

def augmentation_transforms(canvas_width, use_normalized_clip):
  """Image transforms to produce distorted crops to augment the evaluation.

  Args:
    canvas_width: width of the drawing canvas
    use_normalized_clip: Normalisation to better suit CLIP's training data

  Returns:
    transforms
  """
  if use_normalized_clip:
    augment_trans = transforms.Compose(
        [transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
         transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
         transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))])
  else:
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
        transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
    ])

  return augment_trans

# Video creator

class VideoWriter:
  """Create a video from image frames."""

  def __init__(self, filename="_autoplay.mp4", fps=20.0, **kw):
    """Video creator.

    Creates and display a video made from frames. The default
    filename causes the video to be displayed on exit.

    Args:
      filename: name of video file
      fps: frames per second for video
      **kw: args to be passed to FFMPEG_VideoWriter

    Returns:
      VideoWriter instance.
    """

    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    """Add image to video.

    Add new frame to image file, creating VideoWriter if requried.

    Args:
      img: array-like frame, shape [X, Y, 3] or [X, Y]

    Returns:
      None
    """

    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.params["filename"] == "_autoplay.mp4":
      self.show()

  def show(self, **kw):
    """Display video.

    Args:
      **kw: args to be passed to mvp.ipython_display

    Returns:
      None
    """
    self.close()
    fn = self.params["filename"]
    # display(mvp.ipython_display(fn, **kw))


# Training functions

def get_features(prompt, negative_prompt_1, negative_prompt_2):
  # Tokenize prompts and coompute CLIP features.
  text_input = clip.tokenize(prompt).to(device)
  text_input_neg1 = clip.tokenize(negative_prompt_1).to(device)
  text_input_neg2 = clip.tokenize(negative_prompt_2).to(device)
  with torch.no_grad():
    features = clip_model.encode_text(text_input)
    neg1_features = clip_model.encode_text(text_input_neg1)
    neg2_features = clip_model.encode_text(text_input_neg2)
  return features, neg1_features, neg2_features

# Create writers.
def load_torch_img(filename):
  img = PIL.Image.open(filename).convert(mode="RGB")
  img = img.resize((CANVAS_WIDTH, CANVAS_HEIGHT))
  img = np.float32(img)
  img = torch.from_numpy(img).to(torch.float32) / 255.0
  #img = img.pow(2.0)
  img = img.to(pydiffvg.get_device())
  img = img.unsqueeze(0)
  img = img.permute(0, 3, 1, 2)
  return img


def create_generator(grammar_type):
  """Create the drawing generator.

  Args:
    grammar_type: string defining the class of generator to use

  Returns:
    stroke generator instance
  """
  if grammar_type == "Arnheim 2":
    new_generator = CurveNetworkHierarchicalLSTM()
  elif grammar_type == "Photographic":
    new_generator = CurveNetworkPhotographicLSTM()
  elif grammar_type == "DPPN":
    new_generator = CurveNetworkDPPN()
  elif grammar_type == "SEQTOSEQ":
    new_generator = CurveNetworkSEQTOSEQ()
  else:
    print("Unknown drawing function:", grammar_type)

  if LOAD_MODEL:
    state_dict = torch.load(DIR_RESULTS + "/generator.pt")
    new_generator.load_state_dict(state_dict)
    with torch.no_grad():
      for name, param in new_generator.named_parameters():
        if "positions_top" in name:
          print("resetting positions TOP ", flush=True)
          param.copy_(2.0*(torch.rand(BATCH_SIZE, EMBEDDING_SIZE)-0.5))
        if "direct_points" in name:
          print("resetting direct points")
          points_in = []
          for _ in range(BATCH_SIZE * LSTM_ITERATIONS * LSTM_ITERATIONS):
            p0 = (1.5*(random.random()-0.5), 1.5*(random.random()-0.5))
            points_in.append(p0)
            for _ in range(1):
              radius = 0.1
              p1 = (p0[0] + radius * (random.random() - 0.5),
                    p0[1] + radius * (random.random() - 0.5))
              p2 = (p1[0] + radius * (random.random() - 0.5),
                    p1[1] + radius * (random.random() - 0.5))
              p3 = (p2[0] + radius * (random.random() - 0.5),
                    p2[1] + radius * (random.random() - 0.5))
              points_in.append(p1)
              points_in.append(p2)
              points_in.append(p3)
              p0 = p3

          param.copy_(torch.Tensor(
              np.array(points_in).flatten().reshape(
                  (BATCH_SIZE*LSTM_ITERATIONS*LSTM_ITERATIONS, 4, 2))))
  return new_generator

def make_optimizer(generator, learning_rate, input_learing_rate, decay_rate):
  """Make optimizer for generator's parameters.

  Args:
    generator: generator model
    learning_rate: learning rate
    input_learing_rate: learning rate for input
    decay_rate: optional learning ratge decay rate

  Returns:
    optimizer
  """
  if decay_rate is not None:
    optim = torch.optim.SGD(generator.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optim, gamma=decay_rate)
  else:
    my_list = ["positions_top", "positions_top2"]
    params = list(map(
        lambda x: x[1],
        list(filter(lambda kv: kv[0] in my_list,
                    generator.named_parameters()))))
    base_params = list(map(
        lambda x: x[1],
        list(filter(
            lambda kv: kv[0] not in my_list, generator.named_parameters()
            ))))
    lr_scheduler = torch.optim.SGD(
        [{"params": base_params}, {"params": params, "lr": input_learing_rate}],
        lr=learning_rate)
  return lr_scheduler

def step_optimization(t, clip_enc, lr_scheduler, generator, augment_trans,
                      text_features, final_step=False):
  """Do a step of optimization.

  Args:
    t: step count
    clip_enc: model for CLIP encoding
    lr_scheduler: optimizer
    generator: drawing generator to optimise
    augment_trans: transforms for image augmentation
    text_features: tuple with the prompt two negative prompts
    final_step: if True does extras such as saving the model
  """
  # Anneal learning rate (NOTE THIS REDUCES the learning rate for the LSTM
  # whether USE_DECAY is set or not!!!)
  if t == int(OPTIM_STEPS * 0.5):
    for g in lr_scheduler.param_groups:
      g["lr"] = g["lr"] / 2.0
  if t == int(OPTIM_STEPS * 0.75):
    for g in lr_scheduler.param_groups:
      g["lr"] = g["lr"] / 2.0

  # Rebuild the generator.
  t0 = time.time()
  lr_scheduler.zero_grad()

  all_points, all_widths, all_colors, _ = generator()

  if isinstance(generator, CurveNetworkPhotographicLSTM):
    all_widths = all_widths[0:NUM_PATHS]
    all_points = all_points[0:(2 * NUM_PATHS * (NUM_SEGMENTS * 3 + 1))]
    all_points = all_points.view(NUM_PATHS, -1, 2)
    all_points = all_points * (CANVAS_HEIGHT // 2 - 2) + CANVAS_HEIGHT // 2
    all_colors = all_colors[:(NUM_PATHS * 4)].view(NUM_PATHS, 4)
  else:
    all_points = all_points * (CANVAS_HEIGHT // 2 - 2) + CANVAS_HEIGHT // 2

  # Convert points to Bézier curves, widths and colours, and rasterize to img.
  t1 = time.time()
  img = render(all_points, all_widths, all_colors)

  if t % VIDEO_STEPS == 0:
    # Write image to video.
    t2 = time.time()
    video_writer.add(img.cpu().detach().numpy()[0])

  # Compute and add losses after augmenting the image with transforms.
  t3 = time.time()
  img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
  loss = 0
  if not MSE_LOSS:
    img_augs = []
    for n in range(NUM_AUGS):
      img_augs.append(augment_trans(img))
    img_batch = torch.cat(img_augs)
    image_features = clip_enc.encode_image(img_batch)
    for n in range(NUM_AUGS):
      loss -= torch.cosine_similarity(text_features[0],
                                      image_features[n:n+1], dim=1)
      if USE_NEG_PROMPTS:
        loss += torch.cosine_similarity(text_features[1],
                                        image_features[n:n+1], dim=1) * 0.3
        loss += torch.cosine_similarity(text_features[2],
                                        image_features[n:n+1], dim=1) * 0.3
  else:
    loss += (img - target_img).pow(2).mean()

  if WIDTH_DIVERSITY_LOSS:
    width_diversity_loss = torch.std(all_widths)
    print("clip loss = ", loss)
    print("WDL = ", -WDL_COEFFICIENT * width_diversity_loss)
    loss = loss - WDL_COEFFICIENT*width_diversity_loss

  writer.add_scalar("Loss/train", loss, t)
  writer.flush()

  # Backpropagate the gradients.
  t4 = time.time()
  loss.backward()
  # Decay the learning rate.
  lr_scheduler.step()

  # Render the big version.
  t5 = time.time()
  if final_step:
    img_big = render(all_points, all_widths, all_colors,
                     multiplier=MULTIPLIER_BIG_IMAGE)
    img_big = img_big.permute(0, 3, 1, 2)  # NHWC -> NCHW
    show_and_save(img_big, t=t, dpi=300)
    print("Saving model...")
    torch.save(generator.state_dict(), DIR_RESULTS + "/generator.pt")

  # Trace the learning error and images.
  if t % TRACE_EVERY == 0:
    # Send gradients to tensorboard.
    grads = []
    for name, param in generator.named_parameters():
      grads.append(param.grad.view(-1))
      writer.add_histogram(name, param.grad.view(-1), t)
    # Show and trace.
    show_and_save(img, dpi=75)
    if USE_DECAY:
      lr = lr_scheduler.get_last_lr()[0]
    else:
      lr = LEARNING_RATE
    print("Iteration {:3d}, lr {}, rendering loss {:.6f}".format(
        t, lr, loss.item()))

  t6 = time.time()
  if PLOT_DURATIONS:
    print(f"gen_fwd: {t1-t0:.4f}s render: {t2-t1:.4f}s video: {t3-t2:.4f}s clip_loss: {t4-t3:.4f}s bprop: {t5-t4:.4f}s big+trace: {t6-t5:.4f}s")

# Hyperparameters

#@markdown Drawing size for evaluation
CANVAS_WIDTH = 224  #@param {type:"integer"}
CANVAS_HEIGHT = 224  #@param {type:"integer"}

#@markdown Relative size of final large image
MULTIPLIER_BIG_IMAGE = 10  #@param {type:"integer"}

#@markdown Number of augmentations to use in evaluation
NUM_AUGS = 4  #@param {type:"integer"}

#@markdown Extra loss to increase diversity of stroke widths
WIDTH_DIVERSITY_LOSS = False  #@param {type:"boolean"}
WDL_COEFFICIENT = 0.01  #@param {type:"number"}

#@markdown Ought to be True but results are better when False
USE_NORMALIZED_CLIP = False  #@param {type:"boolean"}

#@markdown MSE_LOSS attempts to match an image loaded from Drive
MSE_LOSS = False  #@param {type:"boolean"}
DRIVE_IMAGE_FOR_MSE_LOSS = "circle.png"  #@param {type:"string"}

#@markdown Debugging and monitoring
VERBOSE = False  #@param {type:"boolean"}
PLOT_DURATIONS = False  #@param {type:"boolean"}
VIDEO_STEPS = 10  #@param {type:"integer"}
TRACE_EVERY = 10

# @title Images can be saved on Drive
DIR_RESULTS = "content/"  #@param {type:"string"}
os.makedirs(DIR_RESULTS, exist_ok=True)
print(f"Storing results in {DIR_RESULTS}")


target_img = None
if MSE_LOSS:
  target_img = load_torch_img(
      f"{DIR_RESULTS}/{DRIVE_IMAGE_FOR_MSE_LOSS}")
# show_img(target_img)

LOGS_BASE_DIR = "runs"
os.makedirs(LOGS_BASE_DIR, exist_ok=True)

# Configure Image
# Select drawing grammar
USE_GRAMMAR = "Arnheim 2"  #@param ["Arnheim 2", "Photographic", "DPPN", "SEQTOSEQ"]
# Enter a description of the image, e.g. 'a photorealistic chicken'
PROMPT = args.i  #@param {type:"string"}
# Optional negative prompts to reduce certain image elements or characteristics
USE_NEG_PROMPTS = False  #@param {type:"boolean"}
NEG_PROMPT_1 = "messy"  #@param {type:"string"}
NEG_PROMPT_2 = "cluttered"  #@param {type:"string"}

#@markdown Use a white background instead of black
DRAW_WHITE_BACKGROUND = False  #@param {type:"boolean"}

if USE_GRAMMAR == "Photographic":  # Produces more photorealistic paintings.
  BATCH_SIZE = 1
  LEARNING_RATE = 0.004
  INPUT_LEARNING_RATE = 0.1
  NUM_LSTMS = 3
  INPUT_SPEC_SIZE = 50
  NET_LSTM_HIDDENS = 100
  NET_MLP_HIDDENS = 100
  NUM_STROKE_TYPES = 10
  OUTPUT_SIZE = 8 * NUM_STROKE_TYPES
  NUM_PATHS = 2000
  SEQ_LENGTH = int(NUM_PATHS/NUM_STROKE_TYPES)
  OPTIM_STEPS = 1000
  NUM_SEGMENTS = 1
  USE_DECAY = False
  OUTPUT_COEFF_SYSTEMATICITY = 0.1
  OUTPUT_COEFF_WIDTH = 25.0
  OUTPUT_COEFF_COLOUR = 10.0
  LOAD_MODEL = False
  DECAY_RATE = None

if USE_GRAMMAR == "Arnheim 2":  # Produces hierarchically structured paintings.
  BATCH_SIZE = 200  #30  # 100
  LEARNING_RATE = 0.0004  #0.0008  # 0.0004
  INPUT_LEARNING_RATE = 0.01  #0.05  # 0.01
  NUM_LSTMS = 10  # 10
  EMBEDDING_SIZE = 100
  LSTM_ITERATIONS = 5  #10  # 5
  NET_LSTM_HIDDENS = 250
  NET_MLP_HIDDENS = 250
  OUTPUT_SIZE = 100
  OPTIM_STEPS = 1000
  NUM_SEGMENTS = 1
  USE_DECAY = False
  LEARNABLE_FORCING = False
  GRAMMATICAL_FORCING = 1.0
  USE_DROPOUT = False
  DROPOUT_PROP = 0.2
  YELLOW_ABSOLUTE_POSITIONS_USED = True
  WEIGHT_INITIALIZER = True
  WEIGHT_STD = 0.1
  LOAD_MODEL = False
  DECAY_RATE = None

if USE_GRAMMAR == "DPPN":  # Paintings with a feed forward neural network.
  BATCH_SIZE = 2000
  LEARNING_RATE = 0.004
  INPUT_LEARNING_RATE = 0.02
  EMBEDDING_SIZE = 30
  MLP_LAYERS = 3
  NET_MLP_HIDDENS = 250
  OUTPUT_SIZE = 20
  OPTIM_STEPS = 2000
  NUM_SEGMENTS = 1
  USE_DECAY = False
  LEARNABLE_FORCING = False
  GRAMMATICAL_FORCING = 1.00
  USE_DROPOUT = False
  DROPOUT_PROP = 0.2
  YELLOW_ABSOLUTE_POSITIONS_USED = True
  WEIGHT_INITIALIZER = True
  WEIGHT_STD = 0.1
  LOAD_MODEL = False
  DECAY_RATE = None

if USE_GRAMMAR == "SEQTOSEQ":  # Paintings with a simple language model
  BATCH_SIZE = 30
  SEQ_LENGTH = 20
  LEARNING_RATE = 0.0008
  INPUT_LEARNING_RATE = 0.03
  NUM_STROKES = 20
  NET_LSTM_HIDDENS = 250
  EMBEDDING_SIZE = 13
  INPUT_SIZE = OUTPUT_SIZE = EMBEDDING_SIZE
  OPTIM_STEPS = 2000
  NUM_SEGMENTS = 1
  USE_DECAY = False
  LEARNABLE_FORCING = False
  GRAMMATICAL_FORCING = 1.0
  YELLOW_ABSOLUTE_POSITIONS_USED = True
  WEIGHT_INITIALIZER = True
  WEIGHT_STD = 0.1
  LOAD_MODEL = False
  DECAY_RATE = None

if USE_DECAY:
  DECAY_RATE = 0.999
  LEARNING_RATE = 0.1

# Create Drawing!
# images will appear ending with a final large render
# depending on GPU but expect to wait in the order of ~1 hour

writer = SummaryWriter("logs/tensorboard")
video_writer = VideoWriter()

prompt_features = get_features(PROMPT, NEG_PROMPT_1, NEG_PROMPT_2)
augmentations = augmentation_transforms(CANVAS_WIDTH, USE_NORMALIZED_CLIP)

stroke_generator = create_generator(USE_GRAMMAR)
optimizer = make_optimizer(
    stroke_generator, LEARNING_RATE, INPUT_LEARNING_RATE, DECAY_RATE)
clipping_value = 0.1  # arbitrary value of your choosing
torch.nn.utils.clip_grad_norm(stroke_generator.parameters(), clipping_value)

for step in range(OPTIM_STEPS):
  last_step = step == (OPTIM_STEPS - 1)
  step_optimization(step, clip_model, optimizer, stroke_generator,
                    augmentations, prompt_features, final_step=last_step)

video_writer.close()
# video_writer.show()
