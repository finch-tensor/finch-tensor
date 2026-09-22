Compiling MLIR code:
module {
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

  func.func @main(%_A_19: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_20: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_21: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %__A: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_19[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_19[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_19[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_19[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_19[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_20[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_20[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_14 = arith.index_cast %v_13 : i64 to index
    %v_15 = llvm.extractvalue %_A_20[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_16 = arith.index_cast %v_15 : i64 to index
    %v_17 = llvm.extractvalue %_A_21[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_18 = builtin.unrealized_conversion_cast %v_17 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_19 = llvm.extractvalue %_A_21[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_21[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_22 = arith.index_cast %v_21 : i64 to index
    %v_23 = llvm.extractvalue %__A[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = builtin.unrealized_conversion_cast %v_23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_25 = llvm.extractvalue %__A[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = llvm.extractvalue %__A[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_28 = arith.index_cast %v_27 : i64 to index
    %v_29 = arith.constant 0 : index
    %v_30 = memref.dim %v_24, %v_29 : memref<?xf64>
    %v_31 = arith.constant 1 : index
    scf.for %v_32 = %v_29 to %v_30 step %v_31 {
      %v_33 = arith.constant 0.0 : f64
      memref.store %v_33, %v_24[%v_32] : memref<?xf64>
    }
    scf.for %v_34 = %v_29 to %v_8 step %v_31 {
      %v_35 = llvm.extractvalue %__A[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_36 = arith.index_cast %v_35 : i64 to index
      %v_37 = arith.muli %v_36, %v_34 : index
      scf.for %v_38 = %v_29 to %v_16 step %v_31 {
        %v_39 = arith.muli %v_34, %v_16 : index
        %v_40 = arith.addi %v_39, %v_38 : index
        %v_41 = memref.load %v_4[%v_34] : memref<?xindex>
        %v_42 = arith.addi %v_31, %v_34 : index
        %v_43 = memref.load %v_4[%v_42] : memref<?xindex>
        %v_44 = arith.cmpi slt, %v_41, %v_43 : index
        %v_48, %v_49 = scf.if %v_44 -> (index, index) {
          %v_45 = memref.load %v_6[%v_41] : memref<?xindex>
          %v_46 = arith.subi %v_43, %v_31 : index
          %v_47 = memref.load %v_6[%v_46] : memref<?xindex>
          scf.yield %v_45, %v_47 : index, index
        } else {
          scf.yield %v_31, %v_29 : index, index
        }
        %v_50 = memref.load %v_6[%v_41] : memref<?xindex>
        %v_51 = arith.cmpi slt, %v_50, %v_29 : index
        %v_54 = scf.if %v_51 -> (index) {
          %v_52 = arith.subi %v_43, %v_31 : index
          %v_53 = func.call @scansearch(%v_6, %v_29, %v_41, %v_52) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_53 : index
        } else {
          scf.yield %v_41 : index
        }
        %v_90:3 = scf.while (%v_55 = %v_29, %v_56 = %v_54, %v_57 = %v_48) : (index, index, index) -> (index, index, index) {
          %v_58 = arith.addi %v_31, %v_49 : index
          %v_59 = arith.minsi %v_10, %v_58 : index
          %v_60 = arith.cmpi slt, %v_57, %v_59 : index
          scf.condition(%v_60) %v_55, %v_56, %v_57 : index, index, index
        } do {
          ^bb(%v_55: index, %v_56: index, %v_57: index):
          %v_61 = arith.addi %v_31, %v_57 : index
          %v_62 = arith.minsi %v_61, %v_57 : index
          scf.for %v_63 = %v_55 to %v_62 step %v_31 {
            %v_64 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_65 = arith.index_cast %v_64 : i64 to index
            %v_66 = arith.muli %v_65, %v_63 : index
            %v_67 = arith.addi %v_37, %v_66 : index
            %v_68 = memref.load %v_24[%v_67] : memref<?xf64>
            memref.store %v_68, %v_24[%v_67] : memref<?xf64>
          }
          %v_69 = arith.maxsi %v_55, %v_57 : index
          %v_70 = arith.addi %v_31, %v_57 : index
          scf.for %v_71 = %v_69 to %v_70 step %v_31 {
            %v_72 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_73 = arith.index_cast %v_72 : i64 to index
            %v_74 = arith.muli %v_73, %v_71 : index
            %v_75 = arith.addi %v_37, %v_74 : index
            %v_76 = arith.muli %v_38, %v_22 : index
            %v_77 = arith.addi %v_76, %v_71 : index
            %v_78 = memref.load %v_24[%v_75] : memref<?xf64>
            %v_79 = memref.load %v_2[%v_56] : memref<?xf64>
            %v_80 = memref.load %v_12[%v_40] : memref<?xf64>
            %v_81 = arith.mulf %v_79, %v_80 : f64
            %v_82 = memref.load %v_18[%v_77] : memref<?xf64>
            %v_83 = arith.mulf %v_81, %v_82 : f64
            %v_84 = arith.addf %v_78, %v_83 : f64
            memref.store %v_84, %v_24[%v_75] : memref<?xf64>
          }
          %v_85 = arith.addi %v_31, %v_57 : index
          %v_86 = arith.addi %v_31, %v_56 : index
          %v_87 = arith.cmpi slt, %v_86, %v_43 : index
          %v_89 = scf.if %v_87 -> (index) {
            %v_88 = memref.load %v_6[%v_86] : memref<?xindex>
            scf.yield %v_88 : index
          } else {
            scf.yield %v_10 : index
          }
          scf.yield %v_85, %v_86, %v_89 : index, index, index
        }
        %v_91 = arith.addi %v_31, %v_49 : index
        %v_92 = arith.maxsi %v_29, %v_91 : index
        scf.for %v_93 = %v_92 to %v_10 step %v_31 {
          %v_94 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_95 = arith.index_cast %v_94 : i64 to index
          %v_96 = arith.muli %v_95, %v_93 : index
          %v_97 = arith.addi %v_37, %v_96 : index
          %v_98 = memref.load %v_24[%v_97] : memref<?xf64>
          memref.store %v_98, %v_24[%v_97] : memref<?xf64>
        }
      }
    }
    %v_99 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_100 = llvm.insertvalue %__A, %v_99[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_100, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
