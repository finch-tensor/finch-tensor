def gen_ss(idx_type: str) -> tuple[str, str]:
    val_type = pos_type = idx_type
    name = f"scansearch_{idx_type}"
    p_idx = "%p" if pos_type == "index" else "%p_idx"
    p_cast = (
        ""
        if pos_type == "index"
        else f"%p_idx = arith.index_cast %p : {pos_type} to index\n        "
    )
    m_idx = "%m" if pos_type == "index" else "%m_idx"
    m_cast = (
        ""
        if pos_type == "index"
        else f"%m_idx = arith.index_cast %m : {pos_type} to index\n      "
    )
    pos_pair = f"({pos_type}, {pos_type})"

    code = f"""  func.func @{name}(
    %arr: memref<?x{val_type}>, %x: {val_type},
    %lo: {pos_type}, %hi: {pos_type}
  ) -> {pos_type} attributes {{llvm.emit_c_interface}} {{
    %1 = arith.constant 1 : {pos_type}
    %g:2 = scf.while (%d = %1, %p = %lo) : {pos_pair} -> {pos_pair} {{
      %plt = arith.cmpi slt, %p, %hi : {pos_type}
      %cond = scf.if %plt -> (i1) {{
        {p_cast}%ap = memref.load %arr[{p_idx}] : memref<?x{val_type}>
        %al = arith.cmpi slt, %ap, %x : {val_type}
        scf.yield %al : i1
      }} else {{
        %f = arith.constant false
        scf.yield %f : i1
      }}
      scf.condition(%cond) %d, %p : {pos_type}, {pos_type}
    }} do {{
    ^bb0(%d: {pos_type}, %p: {pos_type}):
      %d2 = arith.shli %d, %1 : {pos_type}
      %p2 = arith.addi %p, %d2 : {pos_type}
      scf.yield %d2, %p2 : {pos_type}, {pos_type}
    }}
    %lo1 = arith.subi %g#1, %g#0 : {pos_type}
    %minp = arith.minsi %g#1, %hi : {pos_type}
    %hi1 = arith.addi %minp, %1 : {pos_type}
    %b:2 = scf.while (%l = %lo1, %h = %hi1) : {pos_pair} -> {pos_pair} {{
      %hm1 = arith.subi %h, %1 : {pos_type}
      %go = arith.cmpi slt, %l, %hm1 : {pos_type}
      scf.condition(%go) %l, %h : {pos_type}, {pos_type}
    }} do {{
    ^bb0(%l: {pos_type}, %h: {pos_type}):
      %diff = arith.subi %h, %l : {pos_type}
      %half = arith.shrsi %diff, %1 : {pos_type}
      %m = arith.addi %l, %half : {pos_type}
      {m_cast}%am = memref.load %arr[{m_idx}] : memref<?x{val_type}>
      %al = arith.cmpi slt, %am, %x : {val_type}
      %l2, %h2 = scf.if %al -> ({pos_type}, {pos_type}) {{
        scf.yield %m, %h : {pos_type}, {pos_type}
      }} else {{
        scf.yield %l, %m : {pos_type}, {pos_type}
      }}
      scf.yield %l2, %h2 : {pos_type}, {pos_type}
    }}
    return %b#1 : {pos_type}
  }}
"""
    return name, code
