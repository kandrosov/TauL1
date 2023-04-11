package tau_pkg is
  subtype Int8 is integer range 0 to 255;
  type Int8Array is array (natural range <>) of Int8;
  type Int8Array2D is array (natural range <>, natural range <>) of Int8;
  type Int8Array3D is array (natural range <>, natural range <>, natural range <>) of Int8;

  function flatten3D(input: Int8Array3D) return Int8Array;
end package;

package body tau_pkg is
  function flatten3D(input: Int8Array3D) return Int8Array is
    variable output: Int8Array(0 to input'length(1) * input'length(2) * input'length(3) - 1);
    variable i: natural := 0;
  begin
    for j in input'range(1) loop
      for k in input'range(2) loop
        for l in input'range(3) loop
          output(i) := input(j, k, l);
          i := i + 1;
        end loop;
      end loop;
    end loop;
    return output;
  end function;

end package body;