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

  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_18: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_15[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_15[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_16[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_15 = llvm.extractvalue %_A_16[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_16 = builtin.unrealized_conversion_cast %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_17 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_18[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_22 = builtin.unrealized_conversion_cast %v_21 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_23 = llvm.extractvalue %_A_18[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = arith.index_cast %v_23 : i64 to index
    %v_25 = llvm.extractvalue %_A_18[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_28 = builtin.unrealized_conversion_cast %v_27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_29 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_30 = arith.index_cast %v_29 : i64 to index
    %v_31 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_32 = arith.index_cast %v_31 : i64 to index
    %v_33 = arith.constant 0 : index
    %v_34 = memref.dim %v_22, %v_33 : memref<?xf64>
    %v_35 = arith.constant 1 : index
    scf.for %v_36 = %v_33 to %v_34 step %v_35 {
      %v_37 = arith.constant 0.0 : f64
      memref.store %v_37, %v_22[%v_36] : memref<?xf64>
    }
    scf.for %v_38 = %v_33 to %v_10 step %v_35 {
      %v_39 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_40 = arith.index_cast %v_39 : i64 to index
      %v_41 = arith.muli %v_40, %v_38 : index
      scf.for %v_42 = %v_33 to %v_8 step %v_35 {
        %v_43 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_44 = arith.index_cast %v_43 : i64 to index
        %v_45 = arith.muli %v_44, %v_42 : index
        %v_46 = arith.addi %v_41, %v_45 : index
        %v_47 = memref.load %v_4[%v_42] : memref<?xindex>
        %v_48 = arith.addi %v_35, %v_42 : index
        %v_49 = memref.load %v_4[%v_48] : memref<?xindex>
        %v_50 = arith.cmpi slt, %v_47, %v_49 : index
        %v_54, %v_55 = scf.if %v_50 -> (index, index) {
          %v_51 = memref.load %v_6[%v_47] : memref<?xindex>
          %v_52 = arith.subi %v_49, %v_35 : index
          %v_53 = memref.load %v_6[%v_52] : memref<?xindex>
          scf.yield %v_51, %v_53 : index, index
        } else {
          scf.yield %v_35, %v_33 : index, index
        }
        %v_56 = memref.load %v_6[%v_47] : memref<?xindex>
        %v_57 = arith.cmpi slt, %v_56, %v_33 : index
        %v_60 = scf.if %v_57 -> (index) {
          %v_58 = arith.subi %v_49, %v_35 : index
          %v_59 = func.call @scansearch(%v_6, %v_33, %v_47, %v_58) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_59 : index
        } else {
          scf.yield %v_47 : index
        }
        %v_77:3 = scf.while (%v_61 = %v_33, %v_62 = %v_60, %v_63 = %v_54) : (index, index, index) -> (index, index, index) {
          %v_64 = arith.addi %v_35, %v_55 : index
          %v_65 = arith.minsi %v_10, %v_64 : index
          %v_66 = arith.cmpi slt, %v_63, %v_65 : index
          scf.condition(%v_66) %v_61, %v_62, %v_63 : index, index, index
        } do {
          ^bb(%v_61: index, %v_62: index, %v_63: index):
          %v_67 = arith.maxsi %v_61, %v_63 : index
          %v_68 = arith.addi %v_35, %v_63 : index
          scf.for %v_69 = %v_67 to %v_68 step %v_35 {
            %v_70 = arith.cmpi eq, %v_69, %v_38 : index
            scf.if %v_70 {
              %v_71 = memref.load %v_2[%v_62] : memref<?xf64>
              memref.store %v_71, %v_22[%v_46] : memref<?xf64>
            }
          }
          %v_72 = arith.addi %v_35, %v_63 : index
          %v_73 = arith.addi %v_35, %v_62 : index
          %v_74 = arith.cmpi slt, %v_73, %v_49 : index
          %v_76 = scf.if %v_74 -> (index) {
            %v_75 = memref.load %v_6[%v_73] : memref<?xindex>
            scf.yield %v_75 : index
          } else {
            scf.yield %v_10 : index
          }
          scf.yield %v_72, %v_73, %v_76 : index, index, index
        }
      }
    }
    %v_78 = memref.dim %v_28, %v_33 : memref<?xf64>
    scf.for %v_79 = %v_33 to %v_78 step %v_35 {
      %v_80 = arith.constant 0.0 : f64
      memref.store %v_80, %v_28[%v_79] : memref<?xf64>
    }
    scf.for %v_81 = %v_33 to %v_18 step %v_35 {
      %v_82 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_83 = arith.index_cast %v_82 : i64 to index
      %v_84 = arith.muli %v_83, %v_81 : index
      scf.for %v_85 = %v_33 to %v_26 step %v_35 {
        %v_86 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_87 = arith.index_cast %v_86 : i64 to index
        %v_88 = arith.muli %v_87, %v_85 : index
        %v_89 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_90 = arith.index_cast %v_89 : i64 to index
        %v_91 = arith.muli %v_90, %v_85 : index
        %v_92 = arith.addi %v_84, %v_91 : index
        %v_93 = memref.load %v_14[%v_81] : memref<?xindex>
        %v_94 = arith.addi %v_35, %v_81 : index
        %v_95 = memref.load %v_14[%v_94] : memref<?xindex>
        %v_96 = arith.cmpi slt, %v_93, %v_95 : index
        %v_100, %v_101 = scf.if %v_96 -> (index, index) {
          %v_97 = memref.load %v_16[%v_93] : memref<?xindex>
          %v_98 = arith.subi %v_95, %v_35 : index
          %v_99 = memref.load %v_16[%v_98] : memref<?xindex>
          scf.yield %v_97, %v_99 : index, index
        } else {
          scf.yield %v_35, %v_33 : index, index
        }
        %v_102 = memref.load %v_16[%v_93] : memref<?xindex>
        %v_103 = arith.cmpi slt, %v_102, %v_33 : index
        %v_106 = scf.if %v_103 -> (index) {
          %v_104 = arith.subi %v_95, %v_35 : index
          %v_105 = func.call @scansearch(%v_16, %v_33, %v_93, %v_104) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_105 : index
        } else {
          scf.yield %v_93 : index
        }
        %v_138:3 = scf.while (%v_107 = %v_33, %v_108 = %v_106, %v_109 = %v_100) : (index, index, index) -> (index, index, index) {
          %v_110 = arith.addi %v_35, %v_101 : index
          %v_111 = arith.minsi %v_20, %v_110 : index
          %v_112 = arith.cmpi slt, %v_109, %v_111 : index
          scf.condition(%v_112) %v_107, %v_108, %v_109 : index, index, index
        } do {
          ^bb_2(%v_107: index, %v_108: index, %v_109: index):
          %v_113 = arith.addi %v_35, %v_109 : index
          %v_114 = arith.minsi %v_113, %v_109 : index
          scf.for %v_115 = %v_107 to %v_114 step %v_35 {
            %v_116 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_117 = arith.index_cast %v_116 : i64 to index
            %v_118 = arith.muli %v_117, %v_115 : index
            %v_119 = arith.addi %v_88, %v_118 : index
            %v_120 = memref.load %v_28[%v_119] : memref<?xf64>
            memref.store %v_120, %v_28[%v_119] : memref<?xf64>
          }
          %v_121 = arith.maxsi %v_107, %v_109 : index
          %v_122 = arith.addi %v_35, %v_109 : index
          scf.for %v_123 = %v_121 to %v_122 step %v_35 {
            %v_124 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_125 = arith.index_cast %v_124 : i64 to index
            %v_126 = arith.muli %v_125, %v_123 : index
            %v_127 = arith.addi %v_88, %v_126 : index
            %v_128 = memref.load %v_28[%v_127] : memref<?xf64>
            %v_129 = memref.load %v_22[%v_92] : memref<?xf64>
            %v_130 = memref.load %v_12[%v_108] : memref<?xf64>
            %v_131 = arith.mulf %v_129, %v_130 : f64
            %v_132 = arith.addf %v_128, %v_131 : f64
            memref.store %v_132, %v_28[%v_127] : memref<?xf64>
          }
          %v_133 = arith.addi %v_35, %v_109 : index
          %v_134 = arith.addi %v_35, %v_108 : index
          %v_135 = arith.cmpi slt, %v_134, %v_95 : index
          %v_137 = scf.if %v_135 -> (index) {
            %v_136 = memref.load %v_16[%v_134] : memref<?xindex>
            scf.yield %v_136 : index
          } else {
            scf.yield %v_20 : index
          }
          scf.yield %v_133, %v_134, %v_137 : index, index, index
        }
        %v_139 = arith.addi %v_35, %v_101 : index
        %v_140 = arith.maxsi %v_33, %v_139 : index
        scf.for %v_141 = %v_140 to %v_20 step %v_35 {
          %v_142 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_143 = arith.index_cast %v_142 : i64 to index
          %v_144 = arith.muli %v_143, %v_141 : index
          %v_145 = arith.addi %v_88, %v_144 : index
          %v_146 = memref.load %v_28[%v_145] : memref<?xf64>
          memref.store %v_146, %v_28[%v_145] : memref<?xf64>
        }
      }
    }
    %v_147 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_148 = llvm.insertvalue %_A_7, %v_147[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_148, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
