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
      %v_92:3 = scf.while (%v_53 = %v_33, %v_54 = %v_52, %v_55 = %v_46) : (index, index, index) -> (index, index, index) {
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
              %v_82 = memref.load %v_22[%v_80] : memref<?xf64>
              %v_83 = memref.load %v_2[%v_54] : memref<?xf64>
              %v_84 = arith.constant 0.0 : f64
              %v_85 = arith.cmpf oeq, %v_83, %v_84 : f64
              %v_86 = arith.select %v_85, %v_82, %v_83 : f64
              memref.store %v_86, %v_22[%v_80] : memref<?xf64>
            }
          }
        }
        %v_87 = arith.addi %v_35, %v_55 : index
        %v_88 = arith.addi %v_35, %v_54 : index
        %v_89 = arith.cmpi slt, %v_88, %v_41 : index
        %v_91 = scf.if %v_89 -> (index) {
          %v_90 = memref.load %v_6[%v_88] : memref<?xindex>
          scf.yield %v_90 : index
        } else {
          scf.yield %v_10 : index
        }
        scf.yield %v_87, %v_88, %v_91 : index, index, index
      }
      %v_93 = arith.addi %v_35, %v_47 : index
      %v_94 = arith.maxsi %v_33, %v_93 : index
      scf.for %v_95 = %v_94 to %v_10 step %v_35 {
        %v_96 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_97 = arith.index_cast %v_96 : i64 to index
        %v_98 = arith.muli %v_97, %v_95 : index
        scf.for %v_99 = %v_33 to %v_8 step %v_35 {
          %v_100 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_101 = arith.index_cast %v_100 : i64 to index
          %v_102 = arith.muli %v_101, %v_99 : index
          %v_103 = arith.addi %v_98, %v_102 : index
        }
      }
    }
    %v_104 = memref.dim %v_28, %v_33 : memref<?xf64>
    scf.for %v_105 = %v_33 to %v_104 step %v_35 {
      %v_106 = arith.constant 0.0 : f64
      memref.store %v_106, %v_28[%v_105] : memref<?xf64>
    }
    scf.for %v_107 = %v_33 to %v_18 step %v_35 {
      %v_108 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_109 = arith.index_cast %v_108 : i64 to index
      %v_110 = arith.muli %v_109, %v_107 : index
      scf.for %v_111 = %v_33 to %v_26 step %v_35 {
        %v_112 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_113 = arith.index_cast %v_112 : i64 to index
        %v_114 = arith.muli %v_113, %v_111 : index
        %v_115 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_116 = arith.index_cast %v_115 : i64 to index
        %v_117 = arith.muli %v_116, %v_111 : index
        %v_118 = arith.addi %v_110, %v_117 : index
        %v_119 = memref.load %v_14[%v_107] : memref<?xindex>
        %v_120 = arith.addi %v_35, %v_107 : index
        %v_121 = memref.load %v_14[%v_120] : memref<?xindex>
        %v_122 = arith.cmpi slt, %v_119, %v_121 : index
        %v_126, %v_127 = scf.if %v_122 -> (index, index) {
          %v_123 = memref.load %v_16[%v_119] : memref<?xindex>
          %v_124 = arith.subi %v_121, %v_35 : index
          %v_125 = memref.load %v_16[%v_124] : memref<?xindex>
          scf.yield %v_123, %v_125 : index, index
        } else {
          scf.yield %v_35, %v_33 : index, index
        }
        %v_128 = memref.load %v_16[%v_119] : memref<?xindex>
        %v_129 = arith.cmpi slt, %v_128, %v_33 : index
        %v_132 = scf.if %v_129 -> (index) {
          %v_130 = arith.subi %v_121, %v_35 : index
          %v_131 = func.call @scansearch(%v_16, %v_33, %v_119, %v_130) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_131 : index
        } else {
          scf.yield %v_119 : index
        }
        %v_164:3 = scf.while (%v_133 = %v_33, %v_134 = %v_132, %v_135 = %v_126) : (index, index, index) -> (index, index, index) {
          %v_136 = arith.addi %v_35, %v_127 : index
          %v_137 = arith.minsi %v_20, %v_136 : index
          %v_138 = arith.cmpi slt, %v_135, %v_137 : index
          scf.condition(%v_138) %v_133, %v_134, %v_135 : index, index, index
        } do {
          ^bb_2(%v_133: index, %v_134: index, %v_135: index):
          %v_139 = arith.addi %v_35, %v_135 : index
          %v_140 = arith.minsi %v_139, %v_135 : index
          scf.for %v_141 = %v_133 to %v_140 step %v_35 {
            %v_142 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_143 = arith.index_cast %v_142 : i64 to index
            %v_144 = arith.muli %v_143, %v_141 : index
            %v_145 = arith.addi %v_114, %v_144 : index
            %v_146 = memref.load %v_28[%v_145] : memref<?xf64>
            memref.store %v_146, %v_28[%v_145] : memref<?xf64>
          }
          %v_147 = arith.maxsi %v_133, %v_135 : index
          %v_148 = arith.addi %v_35, %v_135 : index
          scf.for %v_149 = %v_147 to %v_148 step %v_35 {
            %v_150 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_151 = arith.index_cast %v_150 : i64 to index
            %v_152 = arith.muli %v_151, %v_149 : index
            %v_153 = arith.addi %v_114, %v_152 : index
            %v_154 = memref.load %v_28[%v_153] : memref<?xf64>
            %v_155 = memref.load %v_22[%v_118] : memref<?xf64>
            %v_156 = memref.load %v_12[%v_134] : memref<?xf64>
            %v_157 = arith.mulf %v_155, %v_156 : f64
            %v_158 = arith.addf %v_154, %v_157 : f64
            memref.store %v_158, %v_28[%v_153] : memref<?xf64>
          }
          %v_159 = arith.addi %v_35, %v_135 : index
          %v_160 = arith.addi %v_35, %v_134 : index
          %v_161 = arith.cmpi slt, %v_160, %v_121 : index
          %v_163 = scf.if %v_161 -> (index) {
            %v_162 = memref.load %v_16[%v_160] : memref<?xindex>
            scf.yield %v_162 : index
          } else {
            scf.yield %v_20 : index
          }
          scf.yield %v_159, %v_160, %v_163 : index, index, index
        }
        %v_165 = arith.addi %v_35, %v_127 : index
        %v_166 = arith.maxsi %v_33, %v_165 : index
        scf.for %v_167 = %v_166 to %v_20 step %v_35 {
          %v_168 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_169 = arith.index_cast %v_168 : i64 to index
          %v_170 = arith.muli %v_169, %v_167 : index
          %v_171 = arith.addi %v_114, %v_170 : index
          %v_172 = memref.load %v_28[%v_171] : memref<?xf64>
          memref.store %v_172, %v_28[%v_171] : memref<?xf64>
        }
      }
    }
    %v_173 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_174 = llvm.insertvalue %_A_7, %v_173[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_174, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}
