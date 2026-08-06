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

  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_15[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_15[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_16[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_15 = llvm.extractvalue %_A_16[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = builtin.unrealized_conversion_cast %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_17 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_22 = builtin.unrealized_conversion_cast %v_21 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_23 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = arith.index_cast %v_23 : i64 to index
    %v_25 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = arith.constant 0 : index
    %v_28 = memref.dim %v_22, %v_27 : memref<?xf64>
    %v_29 = arith.constant 1 : index
    scf.for %v_30 = %v_27 to %v_28 step %v_29 {
      %v_31 = arith.constant 0.0 : f64
      memref.store %v_31, %v_22[%v_30] : memref<?xf64>
    }
    scf.for %v_32 = %v_27 to %v_8 step %v_29 {
      %v_33 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_34 = arith.index_cast %v_33 : i64 to index
      %v_35 = arith.muli %v_34, %v_32 : index
      %v_36 = arith.addi %v_27, %v_35 : index
      %v_37 = llvm.extractvalue %_A_15[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_38 = arith.index_cast %v_37 : i64 to index
      %v_39 = arith.muli %v_38, %v_32 : index
      %v_40 = arith.addi %v_27, %v_39 : index
      %v_41 = memref.load %v_4[%v_40] : memref<?xindex>
      %v_42 = arith.addi %v_40, %v_29 : index
      %v_43 = memref.load %v_4[%v_42] : memref<?xindex>
      %v_44 = arith.cmpi slt, %v_41, %v_43 : index
      %v_48, %v_49 = scf.if %v_44 -> (index, index) {
        %v_45 = memref.load %v_6[%v_41] : memref<?xindex>
        %v_46 = arith.subi %v_43, %v_29 : index
        %v_47 = memref.load %v_6[%v_46] : memref<?xindex>
        scf.yield %v_47, %v_45 : index, index
      } else {
        scf.yield %v_27, %v_29 : index, index
      }
      %v_50 = memref.load %v_6[%v_41] : memref<?xindex>
      %v_51 = arith.cmpi slt, %v_50, %v_27 : index
      %v_54 = scf.if %v_51 -> (index) {
        %v_52 = arith.subi %v_43, %v_29 : index
        %v_53 = func.call @scansearch(%v_6, %v_27, %v_41, %v_52) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_53 : index
      } else {
        scf.yield %v_41 : index
      }
      %v_130:3 = scf.while (%v_55 = %v_27, %v_56 = %v_54, %v_57 = %v_49) : (index, index, index) -> (index, index, index) {
        %v_58 = arith.addi %v_48, %v_29 : index
        %v_59 = arith.minsi %v_10, %v_58 : index
        %v_60 = arith.cmpi slt, %v_57, %v_59 : index
        scf.condition(%v_60) %v_55, %v_56, %v_57 : index, index, index
      } do {
        ^bb_2(%v_55: index, %v_56: index, %v_57: index):
        scf.for %v_61 = %v_27 to %v_20 step %v_29 {
          %v_62 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_63 = arith.index_cast %v_62 : i64 to index
          %v_64 = arith.muli %v_63, %v_61 : index
          %v_65 = arith.addi %v_36, %v_64 : index
        }
        %v_66 = arith.maxsi %v_55, %v_57 : index
        %v_67 = arith.addi %v_57, %v_29 : index
        scf.for %v_68 = %v_66 to %v_67 step %v_29 {
          %v_69 = llvm.extractvalue %_A_16[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
          %v_70 = arith.index_cast %v_69 : i64 to index
          %v_71 = arith.muli %v_70, %v_68 : index
          %v_72 = arith.addi %v_27, %v_71 : index
          %v_73 = memref.load %v_14[%v_72] : memref<?xindex>
          %v_74 = arith.addi %v_72, %v_29 : index
          %v_75 = memref.load %v_14[%v_74] : memref<?xindex>
          %v_76 = arith.cmpi slt, %v_73, %v_75 : index
          %v_80, %v_81 = scf.if %v_76 -> (index, index) {
            %v_77 = memref.load %v_16[%v_73] : memref<?xindex>
            %v_78 = arith.subi %v_75, %v_29 : index
            %v_79 = memref.load %v_16[%v_78] : memref<?xindex>
            scf.yield %v_79, %v_77 : index, index
          } else {
            scf.yield %v_27, %v_29 : index, index
          }
          %v_82 = memref.load %v_16[%v_73] : memref<?xindex>
          %v_83 = arith.cmpi slt, %v_82, %v_27 : index
          %v_86 = scf.if %v_83 -> (index) {
            %v_84 = arith.subi %v_75, %v_29 : index
            %v_85 = func.call @scansearch(%v_16, %v_27, %v_73, %v_84) : (memref<?xindex>, index, index, index) -> index
            scf.yield %v_85 : index
          } else {
            scf.yield %v_73 : index
          }
          %v_117:3 = scf.while (%v_87 = %v_27, %v_88 = %v_86, %v_89 = %v_81) : (index, index, index) -> (index, index, index) {
            %v_90 = arith.addi %v_80, %v_29 : index
            %v_91 = arith.minsi %v_20, %v_90 : index
            %v_92 = arith.cmpi slt, %v_89, %v_91 : index
            scf.condition(%v_92) %v_87, %v_88, %v_89 : index, index, index
          } do {
            ^bb(%v_87: index, %v_88: index, %v_89: index):
            %v_93 = arith.addi %v_89, %v_29 : index
            %v_94 = arith.minsi %v_93, %v_89 : index
            scf.for %v_95 = %v_87 to %v_94 step %v_29 {
              %v_96 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
              %v_97 = arith.index_cast %v_96 : i64 to index
              %v_98 = arith.muli %v_97, %v_95 : index
              %v_99 = arith.addi %v_36, %v_98 : index
            }
            %v_100 = arith.maxsi %v_87, %v_89 : index
            %v_101 = arith.addi %v_89, %v_29 : index
            scf.for %v_102 = %v_100 to %v_101 step %v_29 {
              %v_103 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
              %v_104 = arith.index_cast %v_103 : i64 to index
              %v_105 = arith.muli %v_104, %v_102 : index
              %v_106 = arith.addi %v_36, %v_105 : index
              %v_107 = memref.load %v_22[%v_106] : memref<?xf64>
              %v_108 = memref.load %v_2[%v_56] : memref<?xf64>
              %v_109 = memref.load %v_12[%v_88] : memref<?xf64>
              %v_110 = arith.mulf %v_108, %v_109 : f64
              %v_111 = arith.addf %v_107, %v_110 : f64
              memref.store %v_111, %v_22[%v_106] : memref<?xf64>
            }
            %v_112 = arith.addi %v_89, %v_29 : index
            %v_113 = arith.addi %v_88, %v_29 : index
            %v_114 = arith.cmpi slt, %v_113, %v_75 : index
            %v_116 = scf.if %v_114 -> (index) {
              %v_115 = memref.load %v_16[%v_113] : memref<?xindex>
              scf.yield %v_115 : index
            } else {
              scf.yield %v_20 : index
            }
            scf.yield %v_112, %v_113, %v_116 : index, index, index
          }
          %v_118 = arith.addi %v_80, %v_29 : index
          %v_119 = arith.maxsi %v_27, %v_118 : index
          scf.for %v_120 = %v_119 to %v_20 step %v_29 {
            %v_121 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_122 = arith.index_cast %v_121 : i64 to index
            %v_123 = arith.muli %v_122, %v_120 : index
            %v_124 = arith.addi %v_36, %v_123 : index
          }
        }
        %v_125 = arith.addi %v_57, %v_29 : index
        %v_126 = arith.addi %v_56, %v_29 : index
        %v_127 = arith.cmpi slt, %v_126, %v_43 : index
        %v_129 = scf.if %v_127 -> (index) {
          %v_128 = memref.load %v_6[%v_126] : memref<?xindex>
          scf.yield %v_128 : index
        } else {
          scf.yield %v_10 : index
        }
        scf.yield %v_125, %v_126, %v_129 : index, index, index
      }
      scf.for %v_131 = %v_27 to %v_20 step %v_29 {
        %v_132 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_133 = arith.index_cast %v_132 : i64 to index
        %v_134 = arith.muli %v_133, %v_131 : index
        %v_135 = arith.addi %v_36, %v_134 : index
      }
    }
    %v_136 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_137 = llvm.insertvalue %_A_7, %v_136[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_137, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
