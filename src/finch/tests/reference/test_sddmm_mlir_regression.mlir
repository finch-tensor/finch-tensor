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

  func.func @main(%_A_19: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_20: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_21: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %__A: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_19[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_19[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_19[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_19[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_19[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_20[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_20[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = arith.index_cast %v_13 : i64 to index
    %v_15 = llvm.extractvalue %_A_20[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = arith.index_cast %v_15 : i64 to index
    %v_17 = llvm.extractvalue %_A_21[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = builtin.unrealized_conversion_cast %v_17 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_19 = llvm.extractvalue %_A_21[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_21[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
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
      %v_38 = llvm.extractvalue %_A_19[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_39 = arith.index_cast %v_38 : i64 to index
      %v_40 = arith.muli %v_39, %v_34 : index
      %v_41 = llvm.extractvalue %_A_20[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_42 = arith.index_cast %v_41 : i64 to index
      %v_43 = arith.muli %v_42, %v_34 : index
      scf.for %v_44 = %v_29 to %v_16 step %v_31 {
        %v_45 = llvm.extractvalue %_A_20[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_46 = arith.index_cast %v_45 : i64 to index
        %v_47 = arith.muli %v_46, %v_44 : index
        %v_48 = arith.addi %v_43, %v_47 : index
        %v_49 = llvm.extractvalue %_A_21[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_50 = arith.index_cast %v_49 : i64 to index
        %v_51 = arith.muli %v_50, %v_44 : index
        %v_52 = memref.load %v_4[%v_40] : memref<?xindex>
        %v_53 = arith.addi %v_31, %v_40 : index
        %v_54 = memref.load %v_4[%v_53] : memref<?xindex>
        %v_55 = arith.cmpi slt, %v_52, %v_54 : index
        %v_59, %v_60 = scf.if %v_55 -> (index, index) {
          %v_56 = memref.load %v_6[%v_52] : memref<?xindex>
          %v_57 = arith.subi %v_54, %v_31 : index
          %v_58 = memref.load %v_6[%v_57] : memref<?xindex>
          scf.yield %v_58, %v_56 : index, index
        } else {
          scf.yield %v_29, %v_31 : index, index
        }
        %v_61 = memref.load %v_6[%v_52] : memref<?xindex>
        %v_62 = arith.cmpi slt, %v_61, %v_29 : index
        %v_65 = scf.if %v_62 -> (index) {
          %v_63 = arith.subi %v_54, %v_31 : index
          %v_64 = func.call @scansearch(%v_6, %v_29, %v_52, %v_63) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_64 : index
        } else {
          scf.yield %v_52 : index
        }
        %v_103:3 = scf.while (%v_66 = %v_29, %v_67 = %v_65, %v_68 = %v_60) : (index, index, index) -> (index, index, index) {
          %v_69 = arith.addi %v_31, %v_59 : index
          %v_70 = arith.minsi %v_10, %v_69 : index
          %v_71 = arith.cmpi slt, %v_68, %v_70 : index
          scf.condition(%v_71) %v_66, %v_67, %v_68 : index, index, index
        } do {
          ^bb(%v_66: index, %v_67: index, %v_68: index):
          %v_72 = arith.addi %v_31, %v_68 : index
          %v_73 = arith.minsi %v_72, %v_68 : index
          scf.for %v_74 = %v_66 to %v_73 step %v_31 {
            %v_75 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_76 = arith.index_cast %v_75 : i64 to index
            %v_77 = arith.muli %v_76, %v_74 : index
            %v_78 = arith.addi %v_37, %v_77 : index
            %v_79 = memref.load %v_24[%v_78] : memref<?xf64>
            memref.store %v_79, %v_24[%v_78] : memref<?xf64>
          }
          %v_80 = arith.maxsi %v_66, %v_68 : index
          %v_81 = arith.addi %v_31, %v_68 : index
          scf.for %v_82 = %v_80 to %v_81 step %v_31 {
            %v_83 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_84 = arith.index_cast %v_83 : i64 to index
            %v_85 = arith.muli %v_84, %v_82 : index
            %v_86 = arith.addi %v_37, %v_85 : index
            %v_87 = llvm.extractvalue %_A_21[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
            %v_88 = arith.index_cast %v_87 : i64 to index
            %v_89 = arith.muli %v_88, %v_82 : index
            %v_90 = arith.addi %v_51, %v_89 : index
            %v_91 = memref.load %v_24[%v_86] : memref<?xf64>
            %v_92 = memref.load %v_2[%v_67] : memref<?xf64>
            %v_93 = memref.load %v_12[%v_48] : memref<?xf64>
            %v_94 = arith.mulf %v_92, %v_93 : f64
            %v_95 = memref.load %v_18[%v_90] : memref<?xf64>
            %v_96 = arith.mulf %v_94, %v_95 : f64
            %v_97 = arith.addf %v_91, %v_96 : f64
            memref.store %v_97, %v_24[%v_86] : memref<?xf64>
          }
          %v_98 = arith.addi %v_31, %v_68 : index
          %v_99 = arith.addi %v_31, %v_67 : index
          %v_100 = arith.cmpi slt, %v_99, %v_54 : index
          %v_102 = scf.if %v_100 -> (index) {
            %v_101 = memref.load %v_6[%v_99] : memref<?xindex>
            scf.yield %v_101 : index
          } else {
            scf.yield %v_10 : index
          }
          scf.yield %v_98, %v_99, %v_102 : index, index, index
        }
        %v_104 = arith.addi %v_31, %v_59 : index
        %v_105 = arith.maxsi %v_29, %v_104 : index
        scf.for %v_106 = %v_105 to %v_10 step %v_31 {
          %v_107 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_108 = arith.index_cast %v_107 : i64 to index
          %v_109 = arith.muli %v_108, %v_106 : index
          %v_110 = arith.addi %v_37, %v_109 : index
          %v_111 = memref.load %v_24[%v_110] : memref<?xf64>
          memref.store %v_111, %v_24[%v_110] : memref<?xf64>
        }
      }
    }
    %v_112 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_113 = llvm.insertvalue %__A, %v_112[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_113, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
