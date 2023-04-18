library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
library work;
use work.tau_pkg.all;

entity DenseLayerN is
  generic (
    weights: Int8Array2D;
    biases: Int8Array
  );
  port (
    clk: std_logic;
    input: in Int8Array(0 to weights'length(2) - 1);
    output: out Int8Array(0 to weights'length(1) - 1)
  );
end entity;

architecture DenseLayerArch of DenseLayerN is
begin
  process(clk) is
    variable output_v: Int8Array(0 to weights'length(1) - 1);
  begin
    if rising_edge(clk) then
      for i in output_v'range loop
        output_v(i) := biases(i);
        for j in weights'range(2) loop
          output_v(i) := output_v(i) + input(j) * weights(i, j);
        end loop;
      end loop;
      output <= output_v;
    end if;
  end process;
end DenseLayerArch;

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
library work;
use work.tau_pkg.all;

entity DenseLayer1 is
  generic (
    weights: Int8Array;
    biases: Int8
  );
  port (
    clk: std_logic;
    input: in Int8Array(0 to weights'length - 1);
    output: out Int8
  );
end entity;

architecture DenseLayerArch of DenseLayer1 is
begin
  process(clk) is
    variable output_v: Int8;
  begin
    if rising_edge(clk) then
      output_v := biases;
      for j in weights'range loop
        output_v := output_v + input(j) * weights(j);
      end loop;
      output <= output_v;
    end if;
  end process;
end DenseLayerArch;
