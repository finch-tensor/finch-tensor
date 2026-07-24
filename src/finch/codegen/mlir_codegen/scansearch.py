SCANSEARCH = """
  func.func @scansearch(
      %arr: memref<?xindex>, %x: index, %lo: index, %hi: index
  ) -> index attributes {llvm.emit_c_interface} {
        %1 = arith.constant 1 : index
        %g:2 = scf.while (%d = %1, %p = %lo) : (index, index) -> (index, index) {
            %plt = arith.cmpi slt, %p, %hi : index
            %cond = scf.if %plt -> (i1) {
                %ap = memref.load %arr[%p] : memref<?xindex>
                %al = arith.cmpi slt, %ap, %x : index
                scf.yield %al : i1
            } else {
                %f = arith.constant false
                scf.yield %f : i1
            }
            scf.condition(%cond) %d, %p : index, index
        } do {
        ^bb0(%d: index, %p: index):
            %d2 = arith.shli %d, %1 : index
            %p2 = arith.addi %p, %d2 : index
            scf.yield %d2, %p2 : index, index
        }
        %lo1 = arith.subi %g#1, %g#0 : index
        %minp = arith.minsi %g#1, %hi : index
        %hi1 = arith.addi %minp, %1 : index
        %b:2 = scf.while (%l = %lo1, %h = %hi1) : (index, index) -> (index, index) {
            %hm1 = arith.subi %h, %1 : index
            %go = arith.cmpi slt, %l, %hm1 : index
            scf.condition(%go) %l, %h : index, index
        } do {
        ^bb0(%l: index, %h: index):
            %diff = arith.subi %h, %l : index
            %half = arith.shrsi %diff, %1 : index
            %m = arith.addi %l, %half : index
            %am = memref.load %arr[%m] : memref<?xindex>
            %al = arith.cmpi slt, %am, %x : index
            %l2, %h2 = scf.if %al -> (index, index) {
                scf.yield %m, %h : index, index
            } else {
                scf.yield %l, %m : index, index
            }
            scf.yield %l2, %h2 : index, index
        }
        return %b#1 : index
    }
"""
