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
    scf.for %v_38 = %v_33 to %v_8 step %v_35 {
      %v_39 = memref.load %v_4[%v_38] : memref<?xindex>
      %v_40 = arith.addi %v_35, %v_38 : index
      %v_41 = memref.load %v_4[%v_40] : memref<?xindex>
      %v_42 = arith.cmpi slt, %v_39, %v_41 : index
      %v_46, %v_47 = scf.if %v_42 -> (index, index) {
        %v_43 = memref.load %v_6[%v_39] : memref<?xindex>
        %v_44 = arith.subi %v_41, %v_35 : index
        %v_45 = memref.load %v_6[%v_44] : memref<?xindex>
        scf.yield %v_43, %v_45 : index, index
      } else {
        scf.yield %v_35, %v_33 : index, index
      }
      %v_48 = memref.load %v_6[%v_39] : memref<?xindex>
      %v_49 = arith.cmpi slt, %v_48, %v_33 : index
      %v_52 = scf.if %v_49 -> (index) {
        %v_50 = arith.subi %v_41, %v_35 : index
        %v_51 = func.call @scansearch(%v_6, %v_33, %v_39, %v_50) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_51 : index
      } else {
        scf.yield %v_39 : index
      }
      %v_88:3 = scf.while (%v_53 = %v_33, %v_54 = %v_52, %v_55 = %v_46) : (index, index, index) -> (index, index, index) {
        %v_56 = arith.addi %v_35, %v_47 : index
        %v_57 = arith.minsi %v_10, %v_56 : index
        %v_58 = arith.cmpi slt, %v_55, %v_57 : index
        scf.condition(%v_58) %v_53, %v_54, %v_55 : index, index, index
      } do {
        ^bb(%v_53: index, %v_54: index, %v_55: index):
        %v_59 = arith.addi %v_35, %v_55 : index
        %v_60 = arith.minsi %v_59, %v_55 : index
        scf.for %v_61 = %v_53 to %v_60 step %v_35 {
          %v_62 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_63 = arith.index_cast %v_62 : i64 to index
          %v_64 = arith.muli %v_63, %v_61 : index
          scf.for %v_65 = %v_33 to %v_8 step %v_35 {
            %v_66 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_67 = arith.index_cast %v_66 : i64 to index
            %v_68 = arith.muli %v_67, %v_65 : index
            %v_69 = arith.addi %v_64, %v_68 : index
          }
        }
        %v_70 = arith.maxsi %v_53, %v_55 : index
        %v_71 = arith.addi %v_35, %v_55 : index
        scf.for %v_72 = %v_70 to %v_71 step %v_35 {
          %v_73 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_74 = arith.index_cast %v_73 : i64 to index
          %v_75 = arith.muli %v_74, %v_72 : index
          scf.for %v_76 = %v_33 to %v_8 step %v_35 {
            %v_77 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_78 = arith.index_cast %v_77 : i64 to index
            %v_79 = arith.muli %v_78, %v_76 : index
            %v_80 = arith.addi %v_75, %v_79 : index
            %v_81 = arith.cmpi eq, %v_76, %v_38 : index
            scf.if %v_81 {
              %v_82 = memref.load %v_2[%v_54] : memref<?xf64>
              memref.store %v_82, %v_22[%v_80] : memref<?xf64>
            }
          }
        }
        %v_83 = arith.addi %v_35, %v_55 : index
        %v_84 = arith.addi %v_35, %v_54 : index
        %v_85 = arith.cmpi slt, %v_84, %v_41 : index
        %v_87 = scf.if %v_85 -> (index) {
          %v_86 = memref.load %v_6[%v_84] : memref<?xindex>
          scf.yield %v_86 : index
        } else {
          scf.yield %v_10 : index
        }
        scf.yield %v_83, %v_84, %v_87 : index, index, index
      }
      %v_89 = arith.addi %v_35, %v_47 : index
      %v_90 = arith.maxsi %v_33, %v_89 : index
      scf.for %v_91 = %v_90 to %v_10 step %v_35 {
        %v_92 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_93 = arith.index_cast %v_92 : i64 to index
        %v_94 = arith.muli %v_93, %v_91 : index
        scf.for %v_95 = %v_33 to %v_8 step %v_35 {
          %v_96 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_97 = arith.index_cast %v_96 : i64 to index
          %v_98 = arith.muli %v_97, %v_95 : index
          %v_99 = arith.addi %v_94, %v_98 : index
        }
      }
    }
    %v_100 = memref.dim %v_28, %v_33 : memref<?xf64>
    scf.for %v_101 = %v_33 to %v_100 step %v_35 {
      %v_102 = arith.constant 0.0 : f64
      memref.store %v_102, %v_28[%v_101] : memref<?xf64>
    }
    scf.for %v_103 = %v_33 to %v_18 step %v_35 {
      %v_104 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_105 = arith.index_cast %v_104 : i64 to index
      %v_106 = arith.muli %v_105, %v_103 : index
      scf.for %v_107 = %v_33 to %v_26 step %v_35 {
        %v_108 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_109 = arith.index_cast %v_108 : i64 to index
        %v_110 = arith.muli %v_109, %v_107 : index
        %v_111 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_112 = arith.index_cast %v_111 : i64 to index
        %v_113 = arith.muli %v_112, %v_107 : index
        %v_114 = arith.addi %v_106, %v_113 : index
        %v_115 = memref.load %v_14[%v_103] : memref<?xindex>
        %v_116 = arith.addi %v_35, %v_103 : index
        %v_117 = memref.load %v_14[%v_116] : memref<?xindex>
        %v_118 = arith.cmpi slt, %v_115, %v_117 : index
        %v_122, %v_123 = scf.if %v_118 -> (index, index) {
          %v_119 = memref.load %v_16[%v_115] : memref<?xindex>
          %v_120 = arith.subi %v_117, %v_35 : index
          %v_121 = memref.load %v_16[%v_120] : memref<?xindex>
          scf.yield %v_119, %v_121 : index, index
        } else {
          scf.yield %v_35, %v_33 : index, index
        }
        %v_124 = memref.load %v_16[%v_115] : memref<?xindex>
        %v_125 = arith.cmpi slt, %v_124, %v_33 : index
        %v_128 = scf.if %v_125 -> (index) {
          %v_126 = arith.subi %v_117, %v_35 : index
          %v_127 = func.call @scansearch(%v_16, %v_33, %v_115, %v_126) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_127 : index
        } else {
          scf.yield %v_115 : index
        }
        %v_160:3 = scf.while (%v_129 = %v_33, %v_130 = %v_128, %v_131 = %v_122) : (index, index, index) -> (index, index, index) {
          %v_132 = arith.addi %v_35, %v_123 : index
          %v_133 = arith.minsi %v_20, %v_132 : index
          %v_134 = arith.cmpi slt, %v_131, %v_133 : index
          scf.condition(%v_134) %v_129, %v_130, %v_131 : index, index, index
        } do {
          ^bb_2(%v_129: index, %v_130: index, %v_131: index):
          %v_135 = arith.addi %v_35, %v_131 : index
          %v_136 = arith.minsi %v_135, %v_131 : index
          scf.for %v_137 = %v_129 to %v_136 step %v_35 {
            %v_138 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_139 = arith.index_cast %v_138 : i64 to index
            %v_140 = arith.muli %v_139, %v_137 : index
            %v_141 = arith.addi %v_110, %v_140 : index
            %v_142 = memref.load %v_28[%v_141] : memref<?xf64>
            memref.store %v_142, %v_28[%v_141] : memref<?xf64>
          }
          %v_143 = arith.maxsi %v_129, %v_131 : index
          %v_144 = arith.addi %v_35, %v_131 : index
          scf.for %v_145 = %v_143 to %v_144 step %v_35 {
            %v_146 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_147 = arith.index_cast %v_146 : i64 to index
            %v_148 = arith.muli %v_147, %v_145 : index
            %v_149 = arith.addi %v_110, %v_148 : index
            %v_150 = memref.load %v_28[%v_149] : memref<?xf64>
            %v_151 = memref.load %v_22[%v_114] : memref<?xf64>
            %v_152 = memref.load %v_12[%v_130] : memref<?xf64>
            %v_153 = arith.mulf %v_151, %v_152 : f64
            %v_154 = arith.addf %v_150, %v_153 : f64
            memref.store %v_154, %v_28[%v_149] : memref<?xf64>
          }
          %v_155 = arith.addi %v_35, %v_131 : index
          %v_156 = arith.addi %v_35, %v_130 : index
          %v_157 = arith.cmpi slt, %v_156, %v_117 : index
          %v_159 = scf.if %v_157 -> (index) {
            %v_158 = memref.load %v_16[%v_156] : memref<?xindex>
            scf.yield %v_158 : index
          } else {
            scf.yield %v_20 : index
          }
          scf.yield %v_155, %v_156, %v_159 : index, index, index
        }
        %v_161 = arith.addi %v_35, %v_123 : index
        %v_162 = arith.maxsi %v_33, %v_161 : index
        scf.for %v_163 = %v_162 to %v_20 step %v_35 {
          %v_164 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_165 = arith.index_cast %v_164 : i64 to index
          %v_166 = arith.muli %v_165, %v_163 : index
          %v_167 = arith.addi %v_110, %v_166 : index
          %v_168 = memref.load %v_28[%v_167] : memref<?xf64>
          memref.store %v_168, %v_28[%v_167] : memref<?xf64>
        }
      }
    }
    %v_169 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_170 = llvm.insertvalue %_A_7, %v_169[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_170, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
