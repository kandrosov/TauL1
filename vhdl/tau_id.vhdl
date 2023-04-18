library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
library work;
use work.tau_pkg.all;

entity tau_id is
  generic (
    constant InputSizeX: natural := 2;
    constant InputSizeY: natural := 2;
    constant NumberOfInputFeatures: natural := 2
  );
  port (
    clk: std_logic;
    input: in Int8Array3D(0 to InputSizeX - 1, 0 to InputSizeY - 1, 0 to NumberOfInputFeatures - 1);
    output: out Int8
  );
end tau_id;

use work.DenseLayer1;
use work.DenseLayerN;

architecture tau_arch of tau_id is
  constant dense1_n_out: natural := 3;
  constant dense1_weights: Int8Array2D(0 to dense1_n_out - 1, 0 to InputSizeX * InputSizeY * NumberOfInputFeatures - 1)  := (( 1, 2, 3, 4, 5, 6, 7, 8 ), (9, 10, 11, 12, 13, 14, 15, 16), (17, 18, 19, 20, 21, 22, 23, 24));
  constant dense1_biases: Int8Array(0 to dense1_n_out - 1) := (25, 26, 27);
  constant dense2_weights: Int8Array(0 to dense1_n_out - 1)  := ( 28, 29, 30 );
  constant dense2_biases: Int8 := 31;
  signal dense1_output: Int8Array(0 to dense1_n_out - 1);
  signal dense2_output: Int8;
begin
  dense1: entity DenseLayerN
    generic map (
      weights => dense1_weights,
      biases => dense1_biases
    )
    port map (
      clk => clk,
      input => flatten3D(input),
      output => dense1_output
    );
  dense2: entity DenseLayer1
    generic map (
      weights => dense2_weights,
      biases => dense2_biases
    )
    port map (
      clk => clk,
      input => dense1_output,
      output => dense2_output
    );

  output <= dense2_output;
  -- process(clk) is
  -- begin
  --   if rising_edge(clk) then
  --   end if;
  -- end process;
end tau_arch;